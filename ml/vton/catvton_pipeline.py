import os
import time
import requests
import uuid
import shutil

class CatVTONPipeline:
    """
    Cloud-based Virtual Try-On using Hugging Face Spaces API (IDM-VTON model).
    This handles sending local files to the Free Gradio API and downloading the results.
    """
    def __init__(self):
        self.is_loaded = True
        
    def download_image(self, url: str, output_path: str):
        response = requests.get(url)
        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path
        
    def try_on_garment(self, avatar_image_path: str, garment_image_path: str, category: str) -> str:
        """
        Runs the diffusion process via Gradio API.
        """
        try:
            from gradio_client import Client, handle_file
        except ImportError:
            print("gradio_client not installed")
            return avatar_image_path
            
        print(f"[Free Cloud VTON] Sending {category} to HF Space...")
        
        try:
            local_garment_path = garment_image_path
            if garment_image_path.startswith("http"):
                if "/data/" in garment_image_path:
                    parts = garment_image_path.split("/data/")
                    local_garment_path = os.path.join("data", parts[-1])
                else:
                    # It's an external URL (like Unsplash), download it
                    local_garment_path = f"data/vton/temp_dl_{uuid.uuid4().hex[:8]}.jpg"
                    self.download_image(garment_image_path, local_garment_path)
            elif garment_image_path.startswith("/data"):
                parts = garment_image_path.split("/data/")
                local_garment_path = os.path.join("data", parts[-1])
            
            if not os.path.exists(local_garment_path):
                print(f"File not found: {local_garment_path}")
                return avatar_image_path
                
            client = Client("yisol/IDM-VTON")
            
            # Predict using the free HF Space
            result = client.predict(
                dict={
                    "background": handle_file(avatar_image_path),
                    "layers": [],
                    "composite": None
                },
                garm_img=handle_file(local_garment_path),
                garment_des=f"A stylish {category}",
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=20,
                seed=42,
                api_name="/tryon"
            )
            
            # The output is a tuple of (result_image_path, masked_image_path)
            # Result image is saved in a local temp folder by Gradio
            if isinstance(result, tuple) and len(result) > 0:
                result_path = result[0]
            else:
                result_path = result
                
            temp_out = f"data/vton/temp_{uuid.uuid4().hex[:8]}.png"
            os.makedirs("data/vton", exist_ok=True)
            
            # Move from gradio temp to our temp
            shutil.copy(result_path, temp_out)
            return temp_out
            
        except Exception as e:
            print(f"HF Space API failed: {e}")
            return None

    def try_on_outfit(self, avatar_id: str, outfit_items: list, output_dir: str) -> str:
        avatar_path = f"data/avatars/{avatar_id}.jpg"
        
        if not os.path.exists(avatar_path):
            os.makedirs("data/avatars", exist_ok=True)
            with open(avatar_path, "wb") as f:
                f.write(requests.get("https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=400&q=80").content)
                
        tops = [i for i in outfit_items if i.get("category") in ["top", "shirt", "sweater", "t-shirt", "jacket", "coat", "outerwear", "dress"]]
        bottoms = [i for i in outfit_items if i.get("category") in ["bottom", "pants", "jeans", "skirt", "shorts"]]
        
        current_avatar = avatar_path
        hf_success = False
        
        if tops:
            top_garment = tops[0]
            res = self.try_on_garment(current_avatar, top_garment["imageUrl"], "top")
            if res:
                current_avatar = res
                hf_success = True
            
        if bottoms:
            bottom_garment = bottoms[0]
            res = self.try_on_garment(current_avatar, bottom_garment["imageUrl"], "bottom")
            if res:
                current_avatar = res
                hf_success = True
                
        os.makedirs(output_dir, exist_ok=True)
        final_out_path = os.path.join(output_dir, f"{avatar_id}_final_{uuid.uuid4().hex[:6]}.png")
        
        if hf_success and current_avatar != avatar_path and os.path.exists(current_avatar):
            shutil.copy(current_avatar, final_out_path)
            os.remove(current_avatar)
            return f"http://localhost:8000/{final_out_path.replace(chr(92), '/')}"
        else:
            # Fallback: PIL Overlay
            try:
                from PIL import Image
                bg = Image.open(avatar_path).convert("RGBA")
                
                def overlay_image(bg_img, img_url, y_offset, height=200):
                    try:
                        # Fix URL
                        local_path = img_url
                        if img_url.startswith("http") and "/data/" in img_url:
                            local_path = os.path.join("data", img_url.split("/data/")[-1])
                        elif img_url.startswith("http"):
                            local_path = f"data/vton/temp_dl_{uuid.uuid4().hex[:8]}.jpg"
                            self.download_image(img_url, local_path)
                        elif img_url.startswith("/data"):
                            local_path = os.path.join("data", img_url.split("/data/")[-1])
                            
                        fg = Image.open(local_path).convert("RGBA")
                        # resize maintaining aspect ratio
                        w_percent = (height / float(fg.size[1]))
                        w_size = int((float(fg.size[0]) * float(w_percent)))
                        fg = fg.resize((w_size, height), Image.LANCZOS)
                        
                        x_offset = (bg_img.size[0] - fg.size[0]) // 2
                        bg_img.paste(fg, (x_offset, y_offset), fg)
                    except Exception as e:
                        print("Fallback overlay failed for an item:", e)
                
                if tops:
                    overlay_image(bg, tops[0]["imageUrl"], y_offset=100, height=250)
                if bottoms:
                    overlay_image(bg, bottoms[0]["imageUrl"], y_offset=300, height=250)
                    
                bg = bg.convert("RGB")
                bg.save(final_out_path)
                return f"http://localhost:8000/{final_out_path.replace(chr(92), '/')}"
                
            except Exception as e:
                print("Total fallback failed:", e)
                return "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?auto=format&fit=crop&w=800&q=80"        
