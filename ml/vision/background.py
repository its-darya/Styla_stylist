from pathlib import Path
from typing import Literal

from PIL import Image

try:
    from rembg import new_session, remove
except ImportError as e:
    raise ImportError(
        "rembg is required for background removal. Install with: pip install rembg onnxruntime Pillow"
    ) from e

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_SESSIONS: dict[str, object] = {}


class BackgroundRemovalError(RuntimeError):
    pass


def _get_session(model: str):
    if model not in _SESSIONS:
        try:
            _SESSIONS[model] = new_session(model)
        except Exception as e:
            raise BackgroundRemovalError(f"Failed to load rembg model '{model}': {e}") from e
    return _SESSIONS[model]


def remove_background(
    input_path: str | Path,
    output_path: str | Path,
    *,
    background: Literal["transparent", "white"] = "transparent",
    model: str = "isnet-general-use",
    alpha_matting: bool = False,
    force: bool = False,
) -> str:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not input_path.is_file():
        raise BackgroundRemovalError(f"Input path is not a file: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise BackgroundRemovalError(
            f"Unsupported image format '{suffix}'. Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    if background not in ("transparent", "white"):
        raise ValueError("background must be 'transparent' or 'white'")

    if output_path.suffix.lower() != ".png":
        raise BackgroundRemovalError("Output path must have .png extension for transparency support")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        img = Image.open(input_path)
        img.verify()
        img = Image.open(input_path)
        img = img.convert("RGB")
    except Exception as e:
        raise BackgroundRemovalError(f"Invalid or corrupted image '{input_path}': {e}") from e

    try:
        session = _get_session(model)
        result = remove(img, session=session, alpha_matting=alpha_matting)
        if isinstance(result, bytes):
            from io import BytesIO

            result = Image.open(BytesIO(result))
        if result.mode != "RGBA":
            result = result.convert("RGBA")
    except BackgroundRemovalError:
        raise
    except Exception as e:
        raise BackgroundRemovalError(f"Background removal failed: {e}") from e

    try:
        if background == "white":
            bg = Image.new("RGB", result.size, (255, 255, 255))
            bg.paste(result, mask=result.split()[3])
            bg.save(output_path, "PNG")
        else:
            result.save(output_path, "PNG")
    except Exception as e:
        raise BackgroundRemovalError(f"Failed to save output to '{output_path}': {e}") from e

    return str(output_path)
