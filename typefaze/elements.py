"""Procedural "elements": decorations applied identically to every letter of a font.

Training sees a font's letters all carrying the same element (dots at stroke ends, a slash, stripes,
an attached shape, ...), so the model practises "the references all have X, so the new letter gets X".
Four families are held out of training entirely and only used to test whether that skill generalises.
Everything is drawn at 256 px (4x) and area-downsampled to 64 px. Glyphs are uint8, 255 = ink.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.morphology import skeletonize

TRAIN_FAMILIES: tuple[str, ...] = (
    "end_dot", "end_bar", "end_circle", "stripes", "inline", "hollow", "slash", "hbar", "double_bar",
    "attach_circle", "attach_square", "attach_star", "attach_diamond",
)
HELDOUT_FAMILIES: tuple[str, ...] = ("end_triangle", "dashes", "attach_heart", "drop_shadow")

UP = 4
BIG = 64 * UP


@dataclass(frozen=True)
class Recipe:
    elements: tuple[tuple[str, dict], ...] = field(default_factory=tuple)
    marker_width: float | None = None  # stroke width in 64-px pixels, or None

    def describe(self) -> str:
        parts = [name for name, _ in self.elements]
        if self.marker_width:
            parts.insert(0, f"marker{self.marker_width:.1f}")
        return "+".join(parts) or "plain"


PLAIN = Recipe((), None)


# ---------------------------------------------------------------- sampling

def sample_element(family: str, rng: np.random.Generator) -> tuple[str, dict]:
    """Random parameters for one element of the given family (sizes in 64-px pixels)."""
    u = lambda a, b: float(rng.uniform(a, b))  # noqa: E731
    anchor = str(rng.choice(["above", "top_right", "baseline_right"]))
    params = {
        "end_dot": lambda: {"r": u(2.5, 3.2)},
        "end_bar": lambda: {"len": u(6, 8), "w": u(2.3, 2.8)},
        "end_circle": lambda: {"r": u(2.5, 3.2), "w": u(1.0, 1.3)},
        "end_triangle": lambda: {"size": u(5, 6.5)},
        "stripes": lambda: {"angle": u(-60, 60), "pitch": u(4, 6), "w": u(1.5, 2.0)},
        "dashes": lambda: {"spacing": u(6, 8), "gap": u(2.0, 2.8)},
        "inline": lambda: {"w": u(0.8, 1.1)},
        "hollow": lambda: {"w": u(1.0, 1.4)},
        "drop_shadow": lambda: {"dx": u(1.5, 2.5), "dy": u(1.5, 2.5)},
        "slash": lambda: {"angle": u(30, 60), "w": u(1.5, 2.0)},
        "hbar": lambda: {"frac": u(0.35, 0.65), "w": u(1.5, 2.0)},
        "double_bar": lambda: {"frac": u(0.35, 0.6), "gap": u(3, 4), "w": u(1.2, 1.5)},
        **{f"attach_{s}": (lambda: {"size": u(6.0, 7.5), "anchor": anchor})
           for s in ("circle", "square", "star", "diamond", "heart")},
    }
    if family not in params:
        raise ValueError(f"unknown element family: {family}")
    return family, params[family]()


def sample_recipe(rng: np.random.Generator, families: tuple[str, ...] = TRAIN_FAMILIES,
                  marker_p: float = 0.3) -> Recipe:
    """40% no elements, 40% one, 20% two (distinct); marker rendering independently with marker_p."""
    u = rng.random()
    n = 0 if u < 0.4 else 1 if u < 0.8 else 2
    chosen = rng.choice(len(families), size=n, replace=False) if n else []
    elements = tuple(sample_element(families[i], rng) for i in chosen)
    marker = float(rng.uniform(2.5, 4.0)) if rng.random() < marker_p else None
    return Recipe(elements, marker)


# ---------------------------------------------------------------- helpers

def _up(g: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(g).resize((BIG, BIG), Image.BILINEAR))


def _down(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32).reshape(64, UP, 64, UP).mean((1, 3)).round().astype(np.uint8)


def _skeleton_ends(ink64: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sk = skeletonize(ink64)
    nb = ndimage.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - sk
    return sk, np.argwhere(sk & (nb == 1))


def _bbox(ink: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(ink)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _shape(draw: ImageDraw.ImageDraw, kind: str, cx: float, cy: float, s: float) -> None:
    """Draw a filled shape of size s (256-px units) centred at cx, cy."""
    h = s / 2
    if kind == "circle":
        draw.ellipse([cx - h, cy - h, cx + h, cy + h], fill=255)
    elif kind == "square":
        draw.rectangle([cx - h, cy - h, cx + h, cy + h], fill=255)
    elif kind == "diamond":
        draw.polygon([(cx, cy - h), (cx + h, cy), (cx, cy + h), (cx - h, cy)], fill=255)
    elif kind == "star":
        pts = []
        for k in range(10):
            r = h if k % 2 == 0 else h * 0.6  # chunky enough to survive at 64 px
            a = -np.pi / 2 + k * np.pi / 5
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
        draw.polygon(pts, fill=255)
    elif kind == "heart":
        r = h * 0.55
        draw.ellipse([cx - h, cy - h * 0.8, cx - h + 2 * r, cy - h * 0.8 + 2 * r], fill=255)
        draw.ellipse([cx + h - 2 * r, cy - h * 0.8, cx + h, cy - h * 0.8 + 2 * r], fill=255)
        draw.polygon([(cx - h, cy - h * 0.2), (cx + h, cy - h * 0.2), (cx, cy + h)], fill=255)
    elif kind == "triangle":
        draw.polygon([(cx, cy - h), (cx + h, cy + h * 0.8), (cx - h, cy + h * 0.8)], fill=255)


# ---------------------------------------------------------------- per-glyph element drawing

def _marker(g: np.ndarray, width: float) -> np.ndarray:
    """Redraw the letter's centre line with an even round brush. Works at 128 px (2x) for speed."""
    mid = np.asarray(Image.fromarray(g).resize((128, 128), Image.BILINEAR)) > 127
    sk = skeletonize(mid)
    r = max(1, int(round(width)))  # brush radius in 128-px pixels = width/2 at 64 px
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    drawn = ndimage.binary_dilation(sk, structure=(xx ** 2 + yy ** 2) <= r * r).astype(np.float32) * 255
    return drawn.reshape(64, 2, 64, 2).mean((1, 3)).round().astype(np.uint8)


def _apply_one(g: np.ndarray, name: str, p: dict) -> np.ndarray:
    ink = g > 127
    if not ink.any():
        return g
    big = Image.fromarray(_up(g))
    d = ImageDraw.Draw(big)
    S = UP
    x0, y0, x1, y1 = _bbox(ink)

    if name.startswith("end_"):
        # Marks go just beyond each stroke end, along the stroke's direction, offset by the local
        # stroke half-width: drawn on top of the end they would vanish into the stroke.
        sk, ends = _skeleton_ends(ink)
        half = ndimage.distance_transform_edt(ink)
        pts = np.argwhere(sk)
        for y, x in ends:
            near = pts[np.hypot(pts[:, 0] - y, pts[:, 1] - x) <= 4]
            v = np.array([y, x], float) - near.mean(0)
            v = v / (np.linalg.norm(v) or 1.0)
            size = p.get("r", p.get("size", 4) / 2)
            off = half[y, x] + size * 0.6
            cy, cx = (y + 0.5 + v[0] * off) * S, (x + 0.5 + v[1] * off) * S
            if name == "end_dot":
                r = p["r"] * S
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
            elif name == "end_circle":
                r = p["r"] * S
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=255, width=max(1, int(p["w"] * S)))
            elif name == "end_bar":  # across the stroke, like a serif
                h = p["len"] * S / 2
                cy, cx = (y + 0.5 + v[0] * half[y, x]) * S, (x + 0.5 + v[1] * half[y, x]) * S
                d.line([cx - v[0] * h, cy + v[1] * h, cx + v[0] * h, cy - v[1] * h], fill=255,
                       width=max(1, int(p["w"] * S)))
            elif name == "end_triangle":
                _shape(d, "triangle", cx, cy, p["size"] * S)
        return _down(np.asarray(big))

    if name in ("stripes", "dashes", "inline", "hollow"):
        a = np.asarray(big).copy()
        yy, xx = np.mgrid[0:BIG, 0:BIG] / S  # 64-px coordinates
        if name == "stripes":
            th = np.deg2rad(p["angle"])
            phase = (xx * np.sin(th) + yy * np.cos(th)) % p["pitch"]
            a[phase < p["w"]] = 0
        elif name == "hollow":
            inner = ndimage.binary_erosion(a > 127, iterations=max(1, int(round(p["w"] * S))))
            a[inner] = 0
        elif name == "inline":
            sk = skeletonize(a > 127)
            cut = ndimage.binary_dilation(sk, iterations=max(1, int(round(p["w"] * S / 2))))
            a[cut] = 0
        elif name == "dashes":
            sk = skeletonize(ink)
            pts = np.argwhere(sk)
            kept: list[np.ndarray] = []
            for q in pts[np.lexsort((pts[:, 1], pts[:, 0]))]:  # greedy points spaced along the skeleton
                if all(np.hypot(*(q - k)) >= p["spacing"] for k in kept):
                    kept.append(q)
            r = p["gap"] * S
            m = Image.new("L", (BIG, BIG), 0)
            md = ImageDraw.Draw(m)
            for y, x in kept:
                cx, cy = (x + 0.5) * S, (y + 0.5) * S
                md.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
            a[np.asarray(m) > 0] = 0
        return _down(a)

    if name == "drop_shadow":
        a = np.asarray(big).astype(np.int32)
        dx, dy = int(round(p["dx"] * S)), int(round(p["dy"] * S))
        sh = np.zeros_like(a)
        sh[dy:, dx:] = a[: BIG - dy, : BIG - dx]
        hollow_out = (sh > 127) & ~(a > 127)  # shadow shows only outside the letter
        out = a.copy()
        out[hollow_out] = 255
        # keep a 1-px gap between letter and shadow so they read as separate
        gap = ndimage.binary_dilation(a > 127, iterations=S) & ~(a > 127)
        out[gap & hollow_out] = 0
        return _down(out.astype(np.uint8))

    w = max(1, int(p.get("w", 1.5) * S))
    X0, Y0, X1, Y1 = x0 * S, y0 * S, (x1 + 1) * S, (y1 + 1) * S
    if name == "slash":
        cx, cy = (X0 + X1) / 2, (Y0 + Y1) / 2
        L = max(X1 - X0, Y1 - Y0) * 0.75 + 2 * S
        th = np.deg2rad(p["angle"])
        dxl, dyl = L * np.cos(th), L * np.sin(th)
        d.line([cx - dxl, cy + dyl, cx + dxl, cy - dyl], fill=255, width=w)
    elif name == "hbar":
        y = Y0 + p["frac"] * (Y1 - Y0)
        d.line([X0 - 2 * S, y, X1 + 2 * S, y], fill=255, width=w)
    elif name == "double_bar":
        y = Y0 + p["frac"] * (Y1 - Y0)
        for off in (-p["gap"] * S / 2, p["gap"] * S / 2):
            d.line([X0 - 2 * S, y + off, X1 + 2 * S, y + off], fill=255, width=w)
    elif name.startswith("attach_"):
        s = p["size"] * S
        gap = 1.5 * S
        if p["anchor"] == "above":
            cx, cy = (X0 + X1) / 2, Y0 - gap - s / 2
        elif p["anchor"] == "top_right":
            cx, cy = X1 + gap + s / 2, Y0 + s / 2
        else:  # baseline_right
            cx, cy = X1 + gap + s / 2, Y1 - s / 2
        cx = float(np.clip(cx, s / 2, BIG - s / 2))
        cy = float(np.clip(cy, s / 2, BIG - s / 2))
        _shape(d, name.split("_", 1)[1], cx, cy, s)
    else:
        raise ValueError(f"unknown element family: {name}")
    return _down(np.asarray(big))


def apply_recipe(glyphs: np.ndarray, recipe: Recipe) -> np.ndarray:
    """(k,64,64) uint8 -> (k,64,64) uint8. Marker rendering first, then each element in order."""
    if recipe == PLAIN or (not recipe.elements and recipe.marker_width is None):
        return glyphs
    out = []
    for g in glyphs:
        x = _marker(g, recipe.marker_width) if recipe.marker_width else g
        for name, p in recipe.elements:
            x = _apply_one(x, name, p)
        out.append(x)
    return np.stack(out).astype(np.uint8)
