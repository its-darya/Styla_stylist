import os
import time
import requests
import uuid
import shutil
from pathlib import Path


def _hf_token() -> str | None:
    """Optional Hugging Face token for the try-on Spaces.

    Without one every request counts against a small quota shared by all
    anonymous users from this address, which runs out after a handful of
    try-ons. A free account token gives a per-account allowance instead.
    """
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.getenv(name, "").strip()
        if token:
            return token
    return None


def _as_rgb(path: str) -> str:
    """Return a path to an alpha-free copy of the image.

    The Spaces re-encode uploads as JPEG, which cannot hold an alpha channel:
    an RGBA avatar or cut-out fails with "cannot write mode RGBA as JPEG"
    before the model ever runs. Flattening onto white matches how the
    garments are stored anyway.
    """
    from PIL import Image

    try:
        img = Image.open(path)
    except Exception:
        return path
    if img.mode not in ("RGBA", "LA", "P"):
        return path

    img = img.convert("RGBA")
    flat = Image.new("RGB", img.size, (255, 255, 255))
    flat.paste(img, mask=img.split()[3])
    # Keep these throwaway copies out of the wardrobe folders.
    tmp_dir = Path(__file__).resolve().parent.parent.parent / "data" / "vton" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / f"{Path(path).stem}.rgb.png"
    flat.save(out)
    return str(out)


def _space_client(space: str):
    """Build a gradio Client, passing the token under whichever name this
    version accepts — it was renamed from `hf_token` to `token` in
    gradio_client 1.x, and passing the wrong one is a TypeError."""
    import inspect

    from gradio_client import Client

    kwargs = {"verbose": False}
    token = _hf_token()
    if token:
        params = inspect.signature(Client.__init__).parameters
        if "token" in params:
            kwargs["token"] = token
        elif "hf_token" in params:
            kwargs["hf_token"] = token
    return Client(space, **kwargs)

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
        self.last_error: str = ""
        self.applied: list[str] = []
        self.skipped: list[str] = []
        # "ai" when a hosted model produced the image, "preview" when it was
        # drawn locally because the hosted models were unavailable.
        self.mode: str = "ai"

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

    def _prepare_garment(self, garment_image_path: str, category: str) -> str | None:
        """Resolve the garment to a local cut-out file the Spaces can accept."""
        local_garment = self._resolve_local_path(garment_image_path)
        if local_garment is None:
            return None

        from ml.vision.segmentation import extract_garment

        garment_dir = os.path.join(os.path.dirname(os.path.dirname(local_garment)), "garments")
        os.makedirs(garment_dir, exist_ok=True)
        item_id = os.path.splitext(os.path.basename(local_garment))[0]
        segmented = os.path.join(garment_dir, f"{item_id}.png")
        if not os.path.exists(segmented):
            if not extract_garment(local_garment, category, segmented):
                shutil.copyfile(local_garment, segmented)
        return segmented

    # -- providers ------------------------------------------------------
    # Each returns a local path to the generated image, or raises. They are
    # tried in order, because these are free community Spaces: any one of
    # them can be rebuilt with a different API, run out of GPU quota, or
    # simply be asleep, and a single hard-coded Space means try-on is broken
    # until someone notices.

    @staticmethod
    def _idm_vton(person: str, garment: str, region: str, category: str):
        from gradio_client import handle_file

        person, garment = _as_rgb(person), _as_rgb(garment)

        result = _space_client("yisol/IDM-VTON").predict(
            dict={"background": handle_file(person), "layers": [], "composite": None},
            garm_img=handle_file(garment),
            garment_des=f"A stylish {category}",
            is_checked=True,
            is_checked_crop=True,
            denoise_steps=30,
            seed=42,
            api_name="/tryon",
        )
        return result[0] if isinstance(result, (list, tuple)) else result

    @staticmethod
    def _ootdiffusion(person: str, garment: str, region: str, category: str):
        from gradio_client import handle_file

        person, garment = _as_rgb(person), _as_rgb(garment)

        result = _space_client("levihsu/OOTDiffusion").predict(
            vton_img=handle_file(person),
            garm_img=handle_file(garment),
            category=region,                # Upper-body | Lower-body | Dress
            n_samples=1,
            n_steps=20,
            image_scale=2.0,
            seed=-1,
            api_name="/process_dc",
        )
        if isinstance(result, list) and result:
            first = result[0]
            return first.get("image") if isinstance(first, dict) else first
        return None

    @staticmethod
    def _catvton(person: str, garment: str, region: str, category: str):
        from gradio_client import handle_file

        person, garment = _as_rgb(person), _as_rgb(garment)

        cloth_type = {"Upper-body": "upper", "Lower-body": "lower", "Dress": "overall"}[region]
        result = _space_client("zhengchong/CatVTON").predict(
            person_image={"background": handle_file(person), "layers": [], "composite": None},
            cloth_image=handle_file(garment),
            cloth_type=cloth_type,
            num_inference_steps=50,
            guidance_scale=2.5,
            seed=42,
            show_type="result only",
            api_name="/submit_function",
        )
        if isinstance(result, dict):
            return result.get("path") or result.get("url")
        return result

    # Order matters: the model that handles each region best comes first.
    PROVIDERS = {
        "Upper-body": (("IDM-VTON", _idm_vton), ("CatVTON", _catvton), ("OOTDiffusion", _ootdiffusion)),
        "Lower-body": (("OOTDiffusion", _ootdiffusion), ("CatVTON", _catvton)),
        "Dress": (("OOTDiffusion", _ootdiffusion), ("CatVTON", _catvton), ("IDM-VTON", _idm_vton)),
    }

    def try_on_garment(
        self,
        person_image_path: str,
        garment_image_path: str,
        category: str,
    ) -> str | None:
        """Put one garment on the person, trying each provider in turn."""
        category = (category or "").lower()
        if category in ("bottom", "pants", "jeans", "skirt", "shorts"):
            region = "Lower-body"
        elif category == "dress":
            region = "Dress"
        else:
            region = "Upper-body"

        garment = self._prepare_garment(garment_image_path, category)
        if garment is None:
            self.last_error = "garment image could not be read"
            return None

        for name, provider in self.PROVIDERS[region]:
            print(f"[VTON] {region} | trying {name}…")
            try:
                result_path = provider.__func__(person_image_path, garment, region, category)
            except Exception as exc:
                self.last_error = str(exc)
                print(f"[VTON] {name} failed: {str(exc)[:200]}")
                continue

            if not result_path or not os.path.exists(str(result_path)):
                self.last_error = f"{name} returned no image"
                print(f"[VTON] {name} returned nothing usable")
                continue

            base = Path(__file__).resolve().parent.parent.parent
            vton_dir = base / "data" / "vton"
            os.makedirs(str(vton_dir), exist_ok=True)
            temp_out = str(vton_dir / f"temp_{uuid.uuid4().hex[:8]}.png")
            shutil.copy(str(result_path), temp_out)
            print(f"[VTON] {name} succeeded: {temp_out}")
            return temp_out

        return None

    # ------------------------------------------------------------------
    # Face Restoration (Face Swap) post-processing
    # ------------------------------------------------------------------

    def _restore_face(self, original_path: str, generated_path: str) -> str:
        """Put the person's own head back on the generated body.

        Try-on models often redraw the face. The previous approach pasted the
        whole top ~25% of the original photo over the result with a vertical
        gradient, which left a grey band across the chest whenever the person
        was not framed exactly as assumed. Instead, segment the head — hair,
        face, hat, sunglasses — and copy only those pixels, feathered at the
        edge so the join is invisible.

        If segmentation is unavailable the generated image is returned
        untouched, which is better than smearing a band across it.
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("[VTON] cv2/numpy missing — skipping face restoration")
            return generated_path

        orig = cv2.imread(original_path)
        gen = cv2.imread(generated_path)
        if orig is None or gen is None:
            return generated_path
        if orig.shape != gen.shape:
            orig = cv2.resize(orig, (gen.shape[1], gen.shape[0]))

        try:
            from PIL import Image

            from ml.vision.segmentation import segment_labels

            rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            seg = segment_labels(Image.fromarray(rgb))
            head = np.isin(seg, (1, 2, 3, 11))  # hat, hair, sunglasses, face
        except Exception as exc:
            print(f"[VTON] head segmentation failed ({exc}) — skipping face restoration")
            return generated_path

        if head.sum() < 0.001 * head.size:
            print("[VTON] no head found — skipping face restoration")
            return generated_path

        # Feather the mask so the hairline and jaw blend instead of cutting.
        mask = (head.astype(np.float32) * 255).astype(np.uint8)
        blur = max(3, (int(round(min(gen.shape[:2]) * 0.01)) | 1))
        mask = cv2.GaussianBlur(mask, (blur, blur), 0).astype(np.float32)[..., None] / 255.0

        blended = orig.astype(np.float32) * mask + gen.astype(np.float32) * (1.0 - mask)
        cv2.imwrite(generated_path, blended.astype(np.uint8))
        print(f"[VTON] head restored: {generated_path}")
        return generated_path


    # ------------------------------------------------------------------
    # Full outfit try-on (multiple garments applied sequentially)
    # ------------------------------------------------------------------

    def _preview_garment(self, person_path: str, garment_image_path: str,
                         category: str) -> str | None:
        """Draw the garment on locally — no network, no GPU quota."""
        category = (category or "").lower()
        if category in ("bottom", "pants", "jeans", "skirt", "shorts"):
            region = "Lower-body"
        elif category == "dress":
            region = "Dress"
        else:
            region = "Upper-body"

        garment = self._resolve_local_path(garment_image_path)
        if garment is None:
            return None
        try:
            from ml.vton.local_preview import apply_garment

            base = Path(__file__).resolve().parent.parent.parent
            vton_dir = base / "data" / "vton"
            vton_dir.mkdir(parents=True, exist_ok=True)
            out = str(vton_dir / f"preview_{uuid.uuid4().hex[:8]}.png")
            if apply_garment(person_path, garment, region, out):
                print(f"[VTON] {region} | drawn locally (preview): {out}")
                return out
        except Exception as exc:
            print(f"[VTON] local preview failed: {exc}")
        return None

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
        # Dress the person the way you would in real life: the base layer
        # first, then the bottom, then the jacket over the top of both. Each
        # step feeds its result in as the next step's person photo, so the
        # order decides what ends up covering what.
        def of(names):
            return [i for i in outfit_items if i.get("category", "").lower() in names]

        base = of({"top", "shirt", "sweater", "t-shirt", "dress"})
        bottoms = of({"bottom", "pants", "jeans", "skirt", "shorts"})
        outer = of({"jacket", "coat", "outerwear"})

        current_person = avatar_path
        vton_success = False
        self.last_error = ""
        self.applied = []
        self.skipped = []
        self.mode = "ai"

        for item in (base[:1] + bottoms[:1] + outer[:1]):
            label = f"{item.get('color', '')} {item.get('category', 'piece')}".strip()
            category = item.get("category", "top")
            res = self.try_on_garment(current_person, item["imageUrl"], category)
            if res is None:
                # The hosted models are free and shared, so they are regularly
                # out of GPU quota or asleep. Rather than return nothing, draw
                # the garment on locally (ml/vton/local_preview.py).
                res = self._preview_garment(current_person, item["imageUrl"], category)
                if res:
                    self.mode = "preview"
            if res:
                current_person = res
                vton_success = True
                self.applied.append(label)
            else:
                self.skipped.append(label)

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
            
            # The local preview never repaints the face, so it only needs
            # restoring after a generative model has been near it.
            if self.mode == "ai":
                self._restore_face(avatar_path, final_out)
            
            return f"http://localhost:8000/data/vton/{filename}"

        # Nothing came back from the try-on service. Say so, rather than
        # inventing a result: the previous fallbacks pasted flat cut-outs over
        # the photo, and failing that returned a stock photo of an unrelated
        # person, both of which read as a finished try-on to the user.
        detail = self.last_error or ""
        if "quota" in detail.lower():
            # The Hugging Face Space runs on a shared free GPU with a daily
            # allowance; its message names the time it resets.
            hint = "" if _hf_token() else (
                " Set HF_TOKEN in .env to a free Hugging Face access token for a "
                "larger per-account allowance instead of the shared anonymous one."
            )
            raise RuntimeError(
                "The virtual try-on service has used up the free GPU allowance. "
                f"{detail.strip()}{hint}"
            )
        raise RuntimeError(
            "The virtual try-on service did not return a result. It is a free, "
            "shared service and is often busy — please try again in a minute."
            + (f" ({detail.strip()[:160]})" if detail else "")
        )
