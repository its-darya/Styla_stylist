#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.evaluate import threshold_sweep
from ml.vision.classify import ClassifiedItem

LABELS = ("top", "bottom", "dress", "outerwear", "shoes")

DEFAULT_ROWS = [
    ("t1", "top", "top", 0.90),
    ("b1", "bottom", "bottom", 0.85),
    ("b2", "bottom", "bottom", 0.80),
    ("d1", "dress", "dress", 0.90),
    ("d2", "dress", "top", 0.40),
    ("s1", "shoes", "shoes", 0.70),
    ("o1", "outerwear", "outerwear", 0.70),
]


def main():
    parser = argparse.ArgumentParser(
        description="Sweep classification confidence threshold and measure Generate impact"
    )
    parser.add_argument("--csv", help="CSV: item_id,true_category,predicted_category,confidence")
    parser.add_argument("--n-outfits", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.csv:
        with open(args.csv, newline="") as f:
            rows = list(csv.reader(f))
        items = [ClassifiedItem(r[0], r[1], r[2], float(r[3])) for r in rows]
    else:
        items = [ClassifiedItem(*r) for r in DEFAULT_ROWS]

    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = threshold_sweep(
        items, LABELS, thresholds=thresholds, n_outfits=args.n_outfits, seed=args.seed
    )

    print(f"{'thr':>4} {'kept':>4} {'acc':>6} {'macF1':>6} {'valid%':>7}")
    for r in results:
        print(
            f"{r['threshold']:>4.1f} {r['kept']:>4} "
            f"{r['accuracy']:>6.3f} {r['macro_f1']:>6.3f} {r['valid_outfit_rate']:>7.2%}"
        )


if __name__ == "__main__":
    main()
