"""Glyph similarity metrics and a helper for spotting sibling font families."""

from __future__ import annotations

import re

import numpy as np


def l1(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference on uint8 images, scaled to [0, 1]."""
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean() / 255.0)


def _box(x: np.ndarray, k: int = 7) -> np.ndarray:
    """Mean over k x k windows (valid region only), via an integral image."""
    c = np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def ssim(a: np.ndarray, b: np.ndarray, k: int = 7) -> float:
    """Structural similarity (Wang et al. 2004) with a 7x7 box window on uint8 images."""
    x, y = a.astype(np.float64), b.astype(np.float64)
    if np.array_equal(x, y):
        return 1.0
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = _box(x, k), _box(y, k)
    vx = _box(x * x, k) - mx * mx
    vy = _box(y * y, k) - my * my
    cxy = _box(x * y, k) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return float(s.mean())


_SUFFIXES = [
    "ultracondensed", "extracondensed", "semicondensed", "condensed", "ultraexpanded", "extraexpanded",
    "semiexpanded", "expanded", "narrow", "wide", "extended", "compressed", "display", "text", "caption",
    "subhead", "sc", "mono", "unicase", "rounded", "stencil", "inline", "outline", "shadow",
]


def family_base(family: str) -> str:
    """Strip width/optical-size/variant suffixes so siblings share a key: encodesanscondensed -> encodesans."""
    f = family.lower()
    changed = True
    while changed:
        changed = False
        for s in _SUFFIXES:
            if f.endswith(s) and len(f) > len(s) + 2:
                f = f[: -len(s)]
                changed = True
    return re.sub(r"\d+$", "", f)


def element_transfer(generated: np.ndarray, decorated: np.ndarray, plain: np.ndarray) -> float:
    """How much more the generated letters look like the decorated truth than the plain truth.

    Mean over letters of SSIM(gen, decorated) - SSIM(gen, plain). Positive = the element was carried.
    All arrays (n,64,64) uint8, aligned letter by letter.
    """
    return float(np.mean([ssim(g, d) - ssim(g, p) for g, d, p in zip(generated, decorated, plain)]))
