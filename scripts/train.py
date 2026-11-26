"""Train TypeFaze on Kaggle (or anywhere with a GPU).

    python scripts/train.py --data /kaggle/input/typefaze-data --out /kaggle/working/run1 --max-hours 11
    python scripts/train.py ... --resume /kaggle/working/run1/ckpt_latest.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from typefaze.data import GlyphDataset, GlyphDatasetV2, handwriting_weights, to_model_space
from typefaze.elements import HELDOUT_FAMILIES, TRAIN_FAMILIES, Recipe, apply_recipe, sample_element
from typefaze.flow import flow_matching_loss, sample, save_checkpoint
from typefaze.glyphs import CHARS, REF_CHARS, char_id
from typefaze.flow import encode
from typefaze.model import GlyphUNet, StyleEncoder
from typefaze.model_v2 import GlyphUNetV2, RefTokenEncoder


def find_file(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p.detach(), 1 - decay)
    for e, b in zip(ema.buffers(), model.buffers()):
        e.copy_(b)


@torch.no_grad()
def sample_grid(unet, enc, preview: list[tuple[np.ndarray, str]], device, path: Path, steps: int = 12) -> None:
    """Rows = fonts. Each row: a label strip, the 5 reference letters, a gap, then generated A–Z a–z."""
    from PIL import ImageDraw

    unet.eval(), enc.eval()
    ref_ids = [char_id(c) for c in REF_CHARS]
    rows = []
    for g, label in preview:
        batch = {
            "refs": torch.full((1, 8, 1, 64, 64), -1.0),
            "ref_ids": torch.zeros(1, 8, dtype=torch.long),
            "ref_mask": torch.zeros(1, 8, dtype=torch.bool),
        }
        batch["refs"][0, :5, 0] = to_model_space(g[ref_ids])
        batch["ref_ids"][0, :5] = torch.tensor(ref_ids)
        batch["ref_mask"][0, :5] = True
        cond = encode(unet, enc, {k: v.to(device) for k, v in batch.items()}, augment=False)
        cond["style"] = cond["style"].expand(len(CHARS), -1)
        gen = sample(unet, cond, torch.arange(len(CHARS), device=device), steps=steps,
                     generator=torch.Generator(device=device).manual_seed(0))
        gen = ((gen[:, 0].cpu().numpy() + 1) * 127.5).astype(np.uint8)
        tag = Image.new("L", (110, 64), 30)
        ImageDraw.Draw(tag).text((4, 26), label[:18], fill=220)
        gap = np.full((64, 16), 60, np.uint8)
        rows.append(np.concatenate([np.asarray(tag), *g[ref_ids], gap, *gen], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def make_preview(glyphs: np.ndarray, arch: str) -> list[tuple[np.ndarray, str]]:
    """v1: 8 plain fonts. v2: 4 plain, 4 with a training element, 2 with a held-out element (marked *)."""
    if arch == "v1":
        return [(g, "plain") for g in glyphs[:8]]
    rng = np.random.default_rng(0)
    out = [(g, "plain") for g in glyphs[:4]]
    for i, g in enumerate(glyphs[4:8]):
        el = sample_element(TRAIN_FAMILIES[(i * 3) % len(TRAIN_FAMILIES)], rng)
        out.append((apply_recipe(g, Recipe((el,), None)), el[0]))
    for i, g in enumerate(glyphs[8:10]):
        el = sample_element(HELDOUT_FAMILIES[i], rng)
        out.append((apply_recipe(g, Recipe((el,), None)), el[0] + " *"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory containing glyphs.npz and manifest.json (searched recursively)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = no step limit")
    ap.add_argument("--ckpt-minutes", type=float, default=20.0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--arch", choices=["v1", "v2"], default="v1")
    ap.add_argument("--grid-steps", type=int, default=12, help="sampling steps for the preview grid")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    npz, manifest = find_file(data, "glyphs.npz"), find_file(data, "manifest.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.arch == "v2":
        ds = GlyphDatasetV2(str(npz), str(manifest), "train")
        sampler = torch.utils.data.WeightedRandomSampler(handwriting_weights(str(manifest), "train"), len(ds))
    else:
        ds, sampler = GlyphDataset(str(npz), str(manifest), "train"), None
    val = GlyphDataset(str(npz), str(manifest), "val")
    preview = make_preview((val.glyphs if len(val) else ds.glyphs)[:10], args.arch)
    print(f"train fonts: {len(ds)}, val fonts: {len(val)}")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=sampler is None, sampler=sampler, num_workers=args.workers,
                    drop_last=len(ds) >= args.batch, persistent_workers=args.workers > 0)

    if args.arch == "v2":
        unet, enc = GlyphUNetV2().to(device), RefTokenEncoder().to(device)
    else:
        unet, enc = GlyphUNet().to(device), StyleEncoder().to(device)
    params = list(unet.parameters()) + list(enc.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema_unet, ema_enc = copy.deepcopy(unet).eval(), copy.deepcopy(enc).eval()
    step = 0
    config = {k: v for k, v in vars(args).items() if k not in ("resume",)}

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        unet.load_state_dict(ck["unet"]), enc.load_state_dict(ck["encoder"])
        ema_unet.load_state_dict(ck["ema_unet"]), ema_enc.load_state_dict(ck["ema_encoder"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        step = int(ck["step"])
        print(f"resumed from step {step}")

    def checkpoint() -> None:
        save_checkpoint(str(out / "ckpt_latest.pt"), unet, enc, step, config,
                        ema_unet=ema_unet.state_dict(), ema_encoder=ema_enc.state_dict(),
                        optimizer=opt.state_dict(), scaler=scaler.state_dict())
        save_checkpoint(str(out / "ckpt_ema.pt"), ema_unet, ema_enc, step, config)
        sample_grid(ema_unet, ema_enc, preview, device, out / f"samples_{step:07d}.png", steps=args.grid_steps)
        print(f"[{time.strftime('%H:%M:%S')}] checkpoint at step {step}", flush=True)

    start = last_ckpt = time.time()
    running, log = 0.0, open(out / "log.txt", "a")
    done = False
    while not done:
        for batch in dl:
            unet.train(), enc.train()
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = flow_matching_loss(unet, enc, batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            update_ema(ema_unet, unet, 0.999)
            update_ema(ema_enc, enc, 0.999)
            step += 1
            running = 0.98 * running + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 100 == 0:
                msg = json.dumps({"step": step, "loss": round(running, 4), "hours": round((time.time() - start) / 3600, 3)})
                print(msg, flush=True)
                log.write(msg + "\n"), log.flush()
            out_of_time = time.time() - start > args.max_hours * 3600
            out_of_steps = args.max_steps and step >= args.max_steps
            if out_of_time or out_of_steps or time.time() - last_ckpt > args.ckpt_minutes * 60:
                checkpoint()
                last_ckpt = time.time()
            if out_of_time or out_of_steps:
                done = True
                break
    print("finished")


if __name__ == "__main__":
    main()
