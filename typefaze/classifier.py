"""A small 52-way letter classifier used to score legibility of generated glyphs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import to_model_space


class GlyphClassifier(nn.Module):
    def __init__(self, n_classes: int = 52):
        super().__init__()

        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                                 nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))

        # read letters at 32x32: plenty for telling 52 letters apart, and fast enough to train on CPU
        self.net = nn.Sequential(nn.AvgPool2d(2), block(1, 32), block(32, 64), block(64, 96),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(96, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_classifier(npz: str, manifest: str, epochs: int = 3, device: str = "cpu", batch: int = 256) -> GlyphClassifier:
    """Train on every letter of every train-split font. Light shift augmentation for robustness."""
    m = json.loads(Path(manifest).read_text())
    idx = [i for i, f in enumerate(m["fonts"]) if f["split"] == "train"]
    g = np.load(npz)["glyphs"][idx].reshape(-1, 64, 64)  # stays uint8; converted one batch at a time
    y = np.tile(np.arange(52), len(idx))
    model = GlyphClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for _ in range(epochs):
        model.train()
        order = np.random.permutation(len(g))
        for i in range(0, len(order), batch):
            sel = np.sort(order[i:i + batch])
            xb = to_model_space(g[sel])[:, None].to(device)
            yb = torch.from_numpy(y[sel]).to(device)
            dx, dy = np.random.randint(-2, 3, size=2)
            xb = torch.roll(xb, shifts=(int(dy), int(dx)), dims=(2, 3))
            loss = nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def predict(model: GlyphClassifier, glyphs: np.ndarray, device: str = "cpu") -> np.ndarray:
    """glyphs (N,64,64) uint8 -> predicted class ids (N,)."""
    out = []
    for i in range(0, len(glyphs), 512):
        x = to_model_space(glyphs[i:i + 512])[:, None].to(device)
        out.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)
