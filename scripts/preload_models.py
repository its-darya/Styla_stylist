"""Download the models into the image at build time.

Without this the first request after a deploy pays for ~1.5 GB of downloads
and times out. Run during `docker build`, so the weights ship in a layer and
startup only has to load them from disk.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from transformers import (
    AutoModel,
    AutoModelForSemanticSegmentation,
    AutoProcessor,
    SegformerImageProcessor,
)

FASHION_CLIP = "patrickjohncyh/fashion-clip"
SEGFORMER = "mattmdjaga/segformer_b2_clothes"

print(f"[preload] {FASHION_CLIP}")
AutoProcessor.from_pretrained(FASHION_CLIP)
AutoModel.from_pretrained(FASHION_CLIP)

print(f"[preload] {SEGFORMER}")
SegformerImageProcessor.from_pretrained(SEGFORMER)
AutoModelForSemanticSegmentation.from_pretrained(SEGFORMER)

print("[preload] rembg isnet-general-use")
from rembg import new_session

new_session("isnet-general-use")

print("[preload] done")
