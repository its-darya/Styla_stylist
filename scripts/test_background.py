#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.vision.background import BackgroundRemovalError, remove_background


def main():
    parser = argparse.ArgumentParser(description="Remove background from an image using rembg")
    parser.add_argument("--input", "-i", required=True, help="Input image path (JPG/PNG/WEBP/BMP)")
    parser.add_argument("--output", "-o", required=True, help="Output PNG path")
    parser.add_argument(
        "--background",
        choices=["transparent", "white"],
        default="transparent",
        help="Background type (default: transparent)",
    )
    parser.add_argument(
        "--model",
        default="isnet-general-use",
        help="rembg model name (default: isnet-general-use)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output if exists")
    args = parser.parse_args()

    try:
        result = remove_background(
            args.input,
            args.output,
            background=args.background,
            model=args.model,
            force=args.force,
        )
        print(f"Input:      {args.input}")
        print(f"Output:     {result}")
        print(f"Background: {args.background}")
        print(f"Model:      {args.model}")
        print("Done. Open the output PNG to visually verify transparency/white background.")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except BackgroundRemovalError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
