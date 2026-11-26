"""Render google/fonts ofl/ into TypeFaze training data.

    python scripts/build_dataset.py --ofl fonts/ofl --out /kaggle/working/typefaze-data
"""

import argparse
import json

from typefaze.data import build_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ofl", required=True, help="path to google/fonts ofl/ directory")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--handwriting-clip", type=float, default=0.06,
                    help="max fraction of ink outside the box for HANDWRITING fonts (others: 0.02)")
    args = ap.parse_args()
    counts = build_dataset(args.ofl, args.out, progress=True, handwriting_clip=args.handwriting_clip)
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
