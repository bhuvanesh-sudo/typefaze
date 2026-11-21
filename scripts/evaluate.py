"""Evaluate TypeFaze on held-out test fonts against a nearest-existing-font baseline.

    python scripts/evaluate.py --data <dataset dir> --ckpt runs/ckpt_ema.pt --out docs/results.md

For each test font, the model sees only H O n a g and must produce the other 47 letters.
Writes a markdown table to --out and a gallery of the worst fonts (failures.png) beside it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from typefaze.baseline import retrieve
from typefaze.classifier import predict, train_classifier
from typefaze.data import to_model_space
from typefaze.flow import load_checkpoint, sample
from typefaze.glyphs import CHARS, REF_CHARS, char_id
from typefaze.metrics import family_base, l1, ssim
from typefaze.model import GlyphUNet, StyleEncoder


def find(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


@torch.no_grad()
def generate(unet, enc, refs: np.ndarray, ref_ids: list[int], targets: list[int], steps: int, device, seed: int,
             cfg: float = 1.0) -> np.ndarray:
    r = torch.full((1, 8, 1, 64, 64), -1.0)
    r[0, :5, 0] = to_model_space(refs)
    ids = torch.zeros(1, 8, dtype=torch.long)
    ids[0, :5] = torch.tensor(ref_ids)
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, :5] = True
    style = enc(r.to(device), ids.to(device), mask.to(device)).expand(len(targets), -1)
    gen = torch.Generator(device=device).manual_seed(seed)
    out = sample(unet, style, torch.tensor(targets, device=device), steps=steps, cfg=cfg, generator=gen)
    return ((out[:, 0].cpu().numpy() + 1) * 127.5).round().astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-fonts", type=int, default=200)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--classifier-epochs", type=int, default=3)
    ap.add_argument("--cfg", type=float, default=1.0, help="classifier-free guidance strength")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data)
    npz, manifest_path = find(data, "glyphs.npz"), find(data, "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    glyphs = np.load(npz)["glyphs"]
    fonts = manifest["fonts"]
    train_idx = [i for i, f in enumerate(fonts) if f["split"] == "train"]
    test_idx = [i for i, f in enumerate(fonts) if f["split"] == "test"][: args.n_fonts]
    bank = glyphs[train_idx]
    print(f"device {device}; {len(test_idx)} test fonts; baseline bank {len(train_idx)} train fonts")

    t0 = time.time()
    clf = train_classifier(str(npz), str(manifest_path), epochs=args.classifier_epochs, device=device)
    print(f"classifier trained in {time.time() - t0:.0f}s")

    unet, enc = GlyphUNet().to(device).eval(), StyleEncoder().to(device).eval()
    step = load_checkpoint(args.ckpt, unet, enc, map_location=device)

    ref_ids = [char_id(c) for c in REF_CHARS]
    targets = [i for i in range(len(CHARS)) if i not in ref_ids]
    train_bases = {family_base(fonts[i]["family"]) for i in train_idx}

    rows = []
    real_all, gen_all, base_all = [], [], []
    for k, fi in enumerate(test_idx):
        g = glyphs[fi]
        gen = generate(unet, enc, g[ref_ids], ref_ids, targets, args.steps, device, seed=k, cfg=args.cfg)
        nearest = retrieve(g[ref_ids], ref_ids, bank)
        base = bank[nearest][targets]
        real = g[targets]
        rows.append({
            "font": fonts[fi]["file"],
            "family": fonts[fi]["family"],
            "sibling_in_train": family_base(fonts[fi]["family"]) in train_bases,
            "nearest_train_font": fonts[train_idx[nearest]]["file"],
            "gen_ssim": float(np.mean([ssim(a, b) for a, b in zip(gen, real)])),
            "gen_l1": float(np.mean([l1(a, b) for a, b in zip(gen, real)])),
            "base_ssim": float(np.mean([ssim(a, b) for a, b in zip(base, real)])),
            "base_l1": float(np.mean([l1(a, b) for a, b in zip(base, real)])),
        })
        real_all.append(real), gen_all.append(gen), base_all.append(base)
        if (k + 1) % 10 == 0:
            print(f"{k + 1}/{len(test_idx)} fonts, {time.time() - t0:.0f}s", flush=True)

    labels = np.tile(np.array(targets), len(test_idx))
    acc = {name: float((predict(clf, np.concatenate(arrs), device) == labels).mean())
           for name, arrs in (("real", real_all), ("gen", gen_all), ("base", base_all))}

    def mean(key, subset=None):
        sel = [r[key] for r in rows if subset is None or r["sibling_in_train"] == subset]
        return float(np.mean(sel)) if sel else float("nan")

    n_sib = sum(r["sibling_in_train"] for r in rows)
    lines = [
        "# TypeFaze evaluation",
        "",
        f"Checkpoint step {step}. {len(rows)} held-out test fonts (split by family). The model sees only "
        f"**H O n a g** from each font and generates the other **47** letters with {args.steps} Euler steps "
        f"(guidance {args.cfg}). "
        "The baseline picks the training font whose H O n a g are closest (pixel L1) and returns its letters.",
        "",
        "| | Classifier accuracy | Mean L1 ↓ | Mean SSIM ↑ |",
        "|---|---|---|---|",
        f"| Real (ceiling) | {acc['real']:.1%} | 0.000 | 1.000 |",
        f"| TypeFaze | {acc['gen']:.1%} | {mean('gen_l1'):.3f} | {mean('gen_ssim'):.3f} |",
        f"| Nearest-font baseline | {acc['base']:.1%} | {mean('base_l1'):.3f} | {mean('base_ssim'):.3f} |",
        "",
        "- **Classifier accuracy:** a separate 52-way letter classifier, trained on training fonts only, reads each "
        "generated glyph. Higher means more legible. It says nothing about style.",
        "- **L1 / SSIM:** pixel-level and structural similarity to the font's *true* letters. They reward matching "
        "the hidden font exactly, so a plausible-but-different design scores low.",
        "",
        "## Sibling-family check",
        "",
        f"{n_sib} of {len(rows)} test fonts have a sibling family in training (same name after stripping suffixes "
        "like Condensed, Display, SC). Scores split by that:",
        "",
        "| Test fonts | Count | TypeFaze SSIM | Baseline SSIM |",
        "|---|---|---|---|",
        f"| With a training sibling | {n_sib} | {mean('gen_ssim', True):.3f} | {mean('base_ssim', True):.3f} |",
        f"| No training sibling | {len(rows) - n_sib} | {mean('gen_ssim', False):.3f} | {mean('base_ssim', False):.3f} |",
        "",
        "## Worst fonts (by TypeFaze SSIM)",
        "",
        "![failures](failures.png)",
        "",
        "Each pair of rows: the real font's 47 letters, then TypeFaze's attempt.",
        "",
        "| Font | TypeFaze SSIM | Baseline SSIM | Training sibling |",
        "|---|---|---|---|",
    ]
    worst = sorted(range(len(rows)), key=lambda i: rows[i]["gen_ssim"])[:12]
    for i in worst:
        r = rows[i]
        lines.append(f"| {r['font']} | {r['gen_ssim']:.3f} | {r['base_ssim']:.3f} | {'yes' if r['sibling_in_train'] else 'no'} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    (out.parent / "results_per_font.json").write_text(json.dumps(rows, indent=1))

    strips = []
    for i in worst:
        strips.append(np.concatenate(list(real_all[i]), axis=1))
        strips.append(np.concatenate(list(gen_all[i]), axis=1))
        strips.append(np.full((6, 64 * len(targets)), 50, np.uint8))
    Image.fromarray(np.concatenate(strips, axis=0)).save(out.parent / "failures.png")
    print("\n".join(lines[5:10]))


if __name__ == "__main__":
    main()
