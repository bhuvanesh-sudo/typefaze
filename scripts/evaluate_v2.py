"""v2 benchmark: does the model carry consistent elements into new letters, and stay legible?

    python scripts/evaluate_v2.py --data <v2 dataset dir> --ckpt-v2 runs/v2/ckpt_ema.pt \
        --ckpt-v1 runs/ckpt_ema.pt --out docs/v2/results.md --steps 4 --cfg 2.0

For every held-out test font and every condition (plain, each training element, each held-out
element, and the handwriting subset), both models see only the decorated H O n a g and generate the
other 47 letters. Each font gets the same element parameters for both models (seeded by font index).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from typefaze.classifier import predict, train_classifier
from typefaze.data import HANDWRITING, to_model_space
from typefaze.elements import HELDOUT_FAMILIES, PLAIN, TRAIN_FAMILIES, Recipe, apply_recipe, sample_element
from typefaze.flow import encode, load_checkpoint, sample
from typefaze.glyphs import CHARS, REF_CHARS, char_id
from typefaze.metrics import element_transfer, ssim
from typefaze.model import GlyphUNet, StyleEncoder
from typefaze.model_v2 import GlyphUNetV2, RefTokenEncoder

REF_IDS = [char_id(c) for c in REF_CHARS]
TARGETS = [i for i in range(len(CHARS)) if i not in REF_IDS]


def find(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def load_model(path: str, device: str):
    arch = torch.load(path, map_location="cpu", weights_only=False).get("config", {}).get("arch", "v1")
    unet, enc = (GlyphUNetV2(), RefTokenEncoder()) if arch == "v2" else (GlyphUNet(), StyleEncoder())
    load_checkpoint(path, unet, enc, map_location=device)
    return unet.to(device).eval(), enc.to(device).eval()


@torch.no_grad()
def generate(model, refs: np.ndarray, steps: int, cfg: float, seed: int, device: str) -> np.ndarray:
    unet, enc = model
    batch = {"refs": torch.full((1, 8, 1, 64, 64), -1.0), "ref_ids": torch.zeros(1, 8, dtype=torch.long),
             "ref_mask": torch.zeros(1, 8, dtype=torch.bool)}
    batch["refs"][0, :5, 0] = to_model_space(refs)
    batch["ref_ids"][0, :5] = torch.tensor(REF_IDS)
    batch["ref_mask"][0, :5] = True
    cond = encode(unet, enc, {k: v.to(device) for k, v in batch.items()}, augment=False)
    cond["style"] = cond["style"].expand(len(TARGETS), -1)
    out = sample(unet, cond, torch.tensor(TARGETS, device=device), steps=steps, cfg=cfg,
                 generator=torch.Generator(device=device).manual_seed(seed))
    return ((out[:, 0].cpu().numpy() + 1) * 127.5).round().astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt-v2", required=True)
    ap.add_argument("--ckpt-v1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-fonts", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=2.0, help="guidance for v2 (tune on val first)")
    ap.add_argument("--cfg-v1", type=float, default=2.0, help="guidance for v1 (its shipped setting)")
    ap.add_argument("--classifier-epochs", type=int, default=3)
    ap.add_argument("--families", default="", help="comma-separated subset (default: all)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data)
    npz, mpath = find(data, "glyphs.npz"), find(data, "manifest.json")
    fonts = json.loads(mpath.read_text())["fonts"]
    glyphs = np.load(npz)["glyphs"]
    test_idx = [i for i, f in enumerate(fonts) if f["split"] == "test"][: args.n_fonts]
    hw_idx = [i for i in test_idx if fonts[i].get("category") == HANDWRITING]
    fams = tuple(args.families.split(",")) if args.families else TRAIN_FAMILIES + HELDOUT_FAMILIES
    t0 = time.time()
    clf = train_classifier(str(npz), str(mpath), epochs=args.classifier_epochs, device=device)
    models = {"v2": load_model(args.ckpt_v2, device), "v1": load_model(args.ckpt_v1, device)}
    cfgs = {"v2": args.cfg, "v1": args.cfg_v1}
    print(f"{len(test_idx)} test fonts ({len(hw_idx)} handwriting), {len(fams)} element families, device {device}")

    def legibility(gens: list[np.ndarray]) -> float:
        if not gens:
            return float("nan")
        return float((predict(clf, np.concatenate(gens), device) == np.tile(TARGETS, len(gens))).mean())

    def run(name: str, idx: list[int], family: str | None):
        res = {"name": name, "n": len(idx), "gen": {"v1": [], "v2": []}, "dec": [], "plain": []}
        for k, fi in enumerate(idx):
            rec = PLAIN if family is None else Recipe((sample_element(family, np.random.default_rng(fi)),), None)
            dec = apply_recipe(glyphs[fi], rec)
            res["dec"].append(dec[TARGETS])
            res["plain"].append(glyphs[fi][TARGETS])
            for m in ("v2", "v1"):
                res["gen"][m].append(generate(models[m], dec[REF_IDS], args.steps, cfgs[m], k, device))
        res["leg_v2"] = legibility(res["gen"]["v2"])
        res["leg_truth"] = legibility(res["dec"])
        for m in ("v2", "v1"):
            res[f"tr_{m}"] = float(np.mean([element_transfer(g, d, p) for g, d, p in
                                           zip(res["gen"][m], res["dec"], res["plain"])])) if idx else float("nan")
            res[f"ssim_{m}"] = float(np.mean([np.mean([ssim(a, b) for a, b in zip(g, p)])
                                             for g, p in zip(res["gen"][m], res["plain"])])) if idx else float("nan")
        print(f"[{time.time() - t0:.0f}s] {name}: legibility {res['leg_v2']:.1%}, transfer v2 {res['tr_v2']:+.3f} v1 {res['tr_v1']:+.3f}", flush=True)
        return res

    plain = run("Plain", test_idx, None)
    hand = run("Handwriting", hw_idx, None)
    rows = [run(f, test_idx, f) for f in fams]

    def pct(x):
        return "n/a" if np.isnan(x) else f"{x:.1%}"

    def num(x):
        return "n/a" if np.isnan(x) else f"{x:+.3f}"

    lines = ["# TypeFaze v2 evaluation", "",
             f"{plain['n']} held-out test fonts ({hand['n']} handwriting). Each model sees only the (decorated) H O n a g "
             f"and generates the other 47 letters; {args.steps} steps, guidance v2 {args.cfg} / v1 {args.cfg_v1}. "
             "Elements marked * were never seen in training.", "",
             "| Condition | Fonts | Legibility v2 | Legibility truth | Transfer v2 | Transfer v1 | Plain SSIM v2 | Plain SSIM v1 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in [plain, hand] + rows:
        is_plain = r["name"] in ("Plain", "Handwriting")
        star = " *" if r["name"] in HELDOUT_FAMILIES else ""
        tr2, tr1 = ("—", "—") if is_plain else (num(r["tr_v2"]), num(r["tr_v1"]))
        s2, s1 = (f"{r['ssim_v2']:.3f}", f"{r['ssim_v1']:.3f}") if is_plain else ("—", "—")
        lines.append(f"| {r['name']}{star} | {r['n']} | {pct(r['leg_v2'])} | {pct(r['leg_truth'])} | {tr2} | {tr1} | {s2} | {s1} |")

    held = [r for r in rows if r["name"] in HELDOUT_FAMILIES]
    deco = [r for r in rows if not np.isnan(r["leg_v2"])]
    gaps = [r["leg_truth"] - r["leg_v2"] for r in deco]
    held_pos = sum(r["tr_v2"] > 0 for r in held)
    bar = [
        (held_pos >= min(2, len(held)) and len(held) > 0,
         f"held-out element transfer positive for {held_pos} of {len(held)} families (need ≥ 2 of 4)"),
        (plain["leg_v2"] >= 0.90, f"plain-input legibility {pct(plain['leg_v2'])} (need ≥ 90%)"),
        (bool(gaps) and float(np.mean(gaps)) <= 0.05,
         f"decorated legibility vs truth: mean gap {np.mean(gaps) * 100 if gaps else float('nan'):.1f} points, "
         f"worst {max(gaps) * 100 if gaps else float('nan'):.1f} (need mean ≤ 5)"),
        (plain["ssim_v2"] >= plain["ssim_v1"] - 0.02,
         f"plain SSIM v2 {plain['ssim_v2']:.3f} vs v1 {plain['ssim_v1']:.3f} (need ≥ v1 − 0.02)"),
    ]
    lines += ["", "## Success bar (fixed before training)", ""]
    lines += [f"{'PASS' if ok else 'FAIL'} — {msg}" for ok, msg in bar]
    lines += ["", "![transfer gallery](transfer-gallery.png)", "",
              "Gallery: per condition, the decorated references, then v2's first 16 generated letters, then the decorated truth."]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    (out.parent / "results_v2.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k not in ("gen", "dec", "plain")} for r in [plain, hand] + rows], indent=1))

    strips = []
    for r in [plain] + rows:
        if not r["n"]:
            continue
        fi = test_idx[0]
        rec = PLAIN if r["name"] == "Plain" else Recipe((sample_element(r["name"], np.random.default_rng(fi)),), None)
        refs = apply_recipe(glyphs[fi], rec)[REF_IDS]
        tag = Image.new("L", (120, 64), 30)
        ImageDraw.Draw(tag).text((4, 26), r["name"] + (" *" if r["name"] in HELDOUT_FAMILIES else ""), fill=220)
        gap = np.full((64, 12), 70, np.uint8)
        strips.append(np.concatenate([np.asarray(tag), *refs, gap, *r["gen"]["v2"][0][:16], gap, *r["dec"][0][:16]], 1))
    Image.fromarray(np.concatenate(strips, 0)).save(out.parent / "transfer-gallery.png")
    print("\n".join(lines[-8:-4]))


if __name__ == "__main__":
    main()
