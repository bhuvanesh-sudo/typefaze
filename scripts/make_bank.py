"""Pack a subset of train-split fonts for the in-browser nearest-font baseline.

    python scripts/make_bank.py --data <dataset dir> --n 200 --out web/public/bank.bin

Writes bank.bin (N x 52 glyphs, 512 bytes each, 1-bit, CHARS order) and bank.json ({n, names}) beside it.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from typefaze.baseline import pack_bank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    glyphs = np.load(data / "glyphs.npz")["glyphs"]
    train = [i for i, f in enumerate(manifest["fonts"]) if f["split"] == "train"]
    pick = sorted(random.Random(args.seed).sample(train, min(args.n, len(train))))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pack_bank(glyphs[pick]))
    meta = {"n": len(pick), "names": [manifest["fonts"][i]["file"] for i in pick]}
    (out.parent / (out.stem + ".json")).write_text(json.dumps(meta))
    print(f"wrote {len(pick)} fonts, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
