#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from ml.vision.color import dominant_color, extract_dominant_colors, rgb_to_hsv_spec


def main():
    parser = argparse.ArgumentParser(description="Extract dominant colors from an image")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--k", type=int, default=3, help="Number of color clusters")
    parser.add_argument("--min-alpha", type=int, default=128, help="Opaque alpha threshold")
    args = parser.parse_args()

    try:
        img = Image.open(args.input)
        clusters = extract_dominant_colors(img, k=args.k, min_alpha=args.min_alpha)
        print(f"Input: {args.input}")
        for i, c in enumerate(clusters):
            spec = rgb_to_hsv_spec(c.rgb)
            print(
                f"  #{i + 1} rgb={c.rgb} hue={spec.hue:.0f} "
                f"sat={spec.saturation:.2f} val={spec.value:.2f} prop={c.proportion:.2%}"
            )
        print(f"Dominant: {dominant_color(clusters)}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
