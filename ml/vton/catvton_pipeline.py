import os
import time
import requests
import uuid
import shutil
from pathlib import Path

# Category mapping: internal category → Kolors label
KOLORS_CATEGORY_MAP = {
    "top": "Upper body",
    "shirt": "Upper body",
    "t-shirt": "Upper body",
    "sweater": "Upper body",
    "jacket": "Upper body",
    "coat": "Upper body",
    "outerwear": "Upper body",
    "dress": "Dress",
    "bottom": "Lower body",
    "pants": "Lower body",
    "jeans": "Lower body",
    "skirt": "Lower body",
    "shorts": "Lower body",
}


class CatVTONPipeline:
    """
    Cloud-based Virtual Try-On using Kwai-Kolors/Kolors-Virtual-Try-On HF Space.
    Sends local image files to the free Gradio API and downloads results.
    Applies garments sequentially: top first, then bottom (result feeds as next person image).
    """

    def __init__(self):
        self.is_loaded = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def download_image(self, url: str, output_path: str) -> str:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path

    def _resolve_local_path(self, image_path: str) -> str | None:
        """
        Convert whatever path format we receive (URL, /data/… prefix, local path)
        to an actual filesystem path.  Returns None if the file can't be found.
        """
        # Make sure we use BASE_DIR which points to project root
        base = Path(__file__).resolve().parent.parent.parent
        
        if image_path.startswith("http"):
            if "/data/" in image_path:
                # Local file served by FastAPI's StaticFiles mount
                parts = image_path.split("/data/")
                local_path = str(base / "data" / parts[-1])
            else:
                # External URL – download to a temp file
                os.makedirs(str(base / "data" / "vton"), exist_ok=True)
                local_path = str(base / "data" / "vton" / f"temp_dl_{uuid.uuid4().hex[:8]}.jpg")
                try:
                    self.download_image(image_path, local_path)
                except Exception as exc:
                    print(f"[VTON] Failed to download {image_path}: {exc}")
                    return None
        else:
            local_path = image_path

        # If it's a relative path starting with data/
        if local_path.startswith("data/") or local_path.startswith("data\\"):
            local_path = str(base / local_path)

        if not os.path.exists(local_path):
            print(f"[Kolors VTON] File not found: {local_path}")
            return None
        return local_path

    # ------------------------------------------------------------------
    # Core try-on (single garment)
    # ------------------------------------------------------------------

    def try_on_garment(
        self,
        person_image_path: str,
        garment_image_path: str,
        category: str,
    ) -> str | None:
        """
        Call the OOTDiffusion HF Space for a single garment.
        """
        try:
            from gradio_client import Client, handle_file
        except ImportError:
            print("[VTON] gradio_client not installed — pip install gradio_client")
            return None

        # OOTDiffusion expects 'Upper-body', 'Lower-body', 'Dress'
        category = category.lower()
        if category in ["bottom", "pants", "jeans", "skirt", "shorts"]:
            oot_category = "Lower-body"
        elif category in ["dress"]:
            oot_category = "Dress"
        else:
            oot_category = "Upper-body"
            
        print(f"[VTON] category={oot_category} | sending to OOTDiffusion HF Space…")

        local_garment = self._resolve_local_path(garment_image_path)
        if local_garment is None:
            return None

        from ml.vision.segmentation import extract_garment
        import shutil
        
        garment_dir = os.path.join(os.path.dirname(os.path.dirname(local_garment)), "garments")
        os.makedirs(garment_dir, exist_ok=True)
        
        item_id = os.path.splitext(os.path.basename(local_garment))[0]
        segmented_garment = os.path.join(garment_dir, f"{item_id}.png")
        
        # Extract garment if not already extracted
        if not os.path.exists(segmented_garment):
            print(f"[VTON] Extracting garment from {local_garment}...")
            success = extract_garment(local_garment, category, segmented_garment)
            if not success:
                print(f"[VTON] Failed to extract garment, falling back to original image.")
                shutil.copyfile(local_garment, segmented_garment)
        
        # Use segmented garment for VTON
        target_garment = segmented_garment

        try:
            if category == "bottom":
                # IDM-VTON does not support bottoms well, use OOTDiffusion for lower body
                client = Client("neyvre/OOTDiffusion")
                result = client.predict(
                    vton_img=handle_file(person_image_path),
                    garm_img=handle_file(target_garment),
                    category="Lower-body",
                    n_samples=1,
                    n_steps=20,
                    image_scale=2.0,
                    seed=-1,
                    api_name="/process_dc"
                )
                if isinstance(result, list) and len(result) > 0:
                    result_path = result[0].get("image")
                else:
                    result_path = None
            else:
                # Use IDM-VTON for tops and dresses
                client = Client("yisol/IDM-VTON")
                result = client.predict(
                    dict={
                        "background": handle_file(person_image_path),
                        "layers": [],
                        "composite": None
                    },
                    garm_img=handle_file(target_garment),
                    garment_des=f"A stylish {category}",
                    is_checked=True,
                    is_checked_crop=True,
                    denoise_steps=30,
                    seed=42,
                    api_name="/tryon"
                )
                if isinstance(result, (list, tuple)) and len(result) > 0:
                    result_path = result[0]
                else:
                    result_path = result

            if result_path is None or not os.path.exists(str(result_path)):
                print(f"[VTON] Unexpected result path: {result_path}")
                return None

            base = Path(__file__).resolve().parent.parent.parent
            vton_dir = base / "data" / "vton"
            os.makedirs(str(vton_dir), exist_ok=True)
            temp_out = str(vton_dir / f"temp_{uuid.uuid4().hex[:8]}.png")
            shutil.copy(str(result_path), temp_out)
            print(f"[VTON] Success: {temp_out}")
            return temp_out

        except Exception as exc:
            print(f"[VTON] HF Space API failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Face Restoration (Face Swap) post-processing
    # ------------------------------------------------------------------

    def _restore_face(self, original_path: str, generated_path: str) -> str:
        """
        Restores the top portion of the original image (the head/face) onto the generated image.
        This prevents VTON models from altering the user's facial identity when the catalog item contains a human model.
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("[VTON] Missing cv2/numpy for face restoration")
            return generated_path

        orig = cv2.imread(original_path)
        gen = cv2.imread(generated_path)
        if orig is None or gen is None:
            return generated_path
            
        # Ensure identical sizes
        if orig.shape != gen.shape:
            orig = cv2.resize(orig, (gen.shape[1], gen.shape[0]))
            
        ih, iw, _ = orig.shape
        
        # Assume the head occupies the top 22% of the image.
        # We blend from 22% to 28% to smoothly transition into the garment.
        blend_start = int(ih * 0.22)
        blend_end = int(ih * 0.28)
        
        mask = np.zeros((ih, iw, 1), dtype=np.float32)
        
        # Top of the image is 1.0 (fully original)
        mask[:blend_start, :] = 1.0
        
        # Gradient fade on the neck/collar
        for i in range(blend_start, blend_end):
            alpha = 1.0 - ((i - blend_start) / max(1, (blend_end - blend_start)))
            mask[i, :] = alpha
            
        # Blend the images
        result = orig.astype(np.float32) * mask + gen.astype(np.float32) * (1 - mask)
        result = result.astype(np.uint8)
        
        cv2.imwrite(generated_path, result)
        print(f"[VTON] Face successfully restored: {generated_path}")
        return generated_path

    # ------------------------------------------------------------------
    # Full outfit try-on (multiple garments applied sequentially)
    # ------------------------------------------------------------------

    def try_on_outfit(
        self,
        avatar_path: str,
        outfit_items: list,
        output_dir: str,
    ) -> str:
        """
        Apply each garment in the outfit sequentially.
        The result of each try-on step becomes the person image for the next step.

        Args:
            avatar_path:   Local filesystem path to the person/avatar photo.
            outfit_items:  List of dicts with keys 'category' and 'imageUrl'.
            output_dir:    Directory where the final composited image is written.

        Returns:
            Public URL of the final result image (via localhost:8000 static serve),
            or a fallback URL on total failure.
        """
        # Separate and order: tops → dresses → bottoms
        tops = [
            i for i in outfit_items
            if i.get("category", "").lower() in
            ["top", "shirt", "sweater", "t-shirt", "jacket", "coat", "outerwear", "dress"]
        ]
        bottoms = [
            i for i in outfit_items
            if i.get("category", "").lower() in
            ["bottom", "pants", "jeans", "skirt", "shorts"]
        ]

        current_person = avatar_path
        vton_success = False

        # Apply top / dress first
        for item in tops[:1]:  # only first top garment
            res = self.try_on_garment(current_person, item["imageUrl"], item.get("category", "top"))
            if res:
                current_person = res
                vton_success = True

        # Apply bottom next (using previous result as person image)
        for item in bottoms[:1]:
            res = self.try_on_garment(current_person, item["imageUrl"], item.get("category", "bottom"))
            if res:
                current_person = res
                vton_success = True

        os.makedirs(output_dir, exist_ok=True)
        filename = f"result_{uuid.uuid4().hex[:8]}.png"
        final_out = os.path.join(output_dir, filename)

        if vton_success and current_person != avatar_path and os.path.exists(current_person):
            shutil.copy(current_person, final_out)
            # Clean up intermediate temp files
            if current_person != avatar_path:
                try:
                    os.remove(current_person)
                except OSError:
                    pass
            
            # Apply face restoration as a post-processing step
            self._restore_face(avatar_path, final_out)
            
            return f"http://localhost:8000/data/vton/{filename}"

        # ------------------------------------------------------------------
        # Fallback: PIL overlay (no Kolors result available)
        # ------------------------------------------------------------------
        try:
            from PIL import Image

            bg = Image.open(avatar_path).convert("RGBA")

            def overlay_item(bg_img: Image.Image, img_url: str, y_offset: int, height: int = 220):
                local = self._resolve_local_path(img_url)
                if not local:
                    return
                try:
                    fg = Image.open(local).convert("RGBA")
                    ratio = height / float(fg.size[1])
                    new_w = int(fg.size[0] * ratio)
                    fg = fg.resize((new_w, height), Image.LANCZOS)
                    x_off = (bg_img.size[0] - new_w) // 2
                    bg_img.paste(fg, (x_off, y_offset), fg)
                except Exception as exc:
                    print(f"[Kolors VTON] PIL overlay failed: {exc}")

            if tops:
                overlay_item(bg, tops[0]["imageUrl"], y_offset=80, height=260)
            if bottoms:
                overlay_item(bg, bottoms[0]["imageUrl"], y_offset=320, height=260)

            bg = bg.convert("RGB")
            bg.save(final_out)
            return f"http://localhost:8000/data/vton/{filename}"

        except Exception as exc:
            print(f"[Kolors VTON] PIL fallback failed: {exc}")
            return "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?auto=format&fit=crop&w=800&q=80"
