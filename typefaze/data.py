"""Dataset building (google/fonts ofl/ -> npz) and the training Dataset."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .glyphs import CHARS, REF_CHARS, char_id
from .render import render_font

FONT_EXTS = {".ttf", ".otf"}
MAX_REFS = 8


def split_families(families: list[str], seed: int = 0) -> dict[str, list[str]]:
    """Deterministic 90/5/5 split of family names (input order does not matter)."""
    fams = sorted(set(families))
    random.Random(seed).shuffle(fams)
    n_val = round(0.05 * len(fams))
    n_test = round(0.05 * len(fams))
    return {
        "val": sorted(fams[:n_val]),
        "test": sorted(fams[n_val:n_val + n_test]),
        "train": sorted(fams[n_val + n_test:]),
    }


def build_dataset(ofl_dir: str, out_dir: str, progress: bool = False) -> dict:
    """Render every font under ofl_dir/<family>/ and write glyphs.npz + manifest.json."""
    root = Path(ofl_dir)
    fam_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    splits = split_families([p.name for p in fam_dirs])
    split_of = {f: s for s, fs in splits.items() for f in fs}

    files = [(d.name, f) for d in fam_dirs for f in sorted(d.iterdir()) if f.suffix.lower() in FONT_EXTS]
    if progress:
        from tqdm import tqdm

        files = tqdm(files, desc="rendering")

    glyphs, fonts, skipped = [], [], []
    for family, f in files:
        g = render_font(str(f))
        if g is None:
            skipped.append(str(f.relative_to(root)))
            continue
        glyphs.append(g)
        fonts.append({"file": str(f.relative_to(root)), "family": family, "split": split_of[family]})

    families = sorted({f["family"] for f in fonts})
    fam_index = {f: i for i, f in enumerate(families)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arr = np.stack(glyphs) if glyphs else np.zeros((0, len(CHARS), 64, 64), np.uint8)
    np.savez_compressed(
        out / "glyphs.npz",
        glyphs=arr,
        family_idx=np.array([fam_index[f["family"]] for f in fonts], dtype=np.int64),
    )
    counts = {
        "kept": len(fonts),
        "skipped": len(skipped),
        **{f"{s}_fonts": sum(f["split"] == s for f in fonts) for s in ("train", "val", "test")},
    }
    manifest = {"fonts": fonts, "families": families, "counts": counts, "skipped_files": skipped}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return counts


def to_model_space(a: np.ndarray) -> torch.Tensor:
    """uint8 (0 paper, 255 ink) -> float in [-1, 1] with ink = +1."""
    return torch.from_numpy(a.astype(np.float32) / 127.5 - 1.0)


class GlyphDataset(Dataset):
    """One item = one font: some reference letters and one target letter to generate."""

    def __init__(self, npz_path: str, manifest_path: str, split: str,
                 k_range: tuple[int, int] = (3, 8), fixed_ref_prob: float = 0.3):
        manifest = json.loads(Path(manifest_path).read_text())
        idx = [i for i, f in enumerate(manifest["fonts"]) if f["split"] == split]
        self.glyphs = np.load(npz_path)["glyphs"][idx]
        self.k_range = k_range
        self.fixed_ref_prob = fixed_ref_prob
        self.fixed_ids = [char_id(c) for c in REF_CHARS]

    def __len__(self) -> int:
        return len(self.glyphs)

    def __getitem__(self, i: int) -> dict:
        n = len(CHARS)
        if random.random() < self.fixed_ref_prob:
            ref_ids = list(self.fixed_ids)
            target = random.choice([c for c in range(n) if c not in ref_ids])
        else:
            target = random.randrange(n)
            k = random.randint(*self.k_range)
            ref_ids = random.sample([c for c in range(n) if c != target], k)

        g = self.glyphs[i]
        refs = torch.full((MAX_REFS, 1, 64, 64), -1.0)
        refs[:len(ref_ids), 0] = to_model_space(g[ref_ids])
        ids = torch.zeros(MAX_REFS, dtype=torch.long)
        ids[:len(ref_ids)] = torch.tensor(ref_ids)
        mask = torch.zeros(MAX_REFS, dtype=torch.bool)
        mask[:len(ref_ids)] = True
        return {
            "refs": refs,
            "ref_ids": ids,
            "ref_mask": mask,
            "target": to_model_space(g[target])[None],
            "target_id": torch.tensor(target, dtype=torch.long),
        }
