"""Render a font file into 52 glyph bitmaps using one shared em/baseline convention."""

from __future__ import annotations

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from .glyphs import BASELINE_ROW, CENTER_COL, CHARS, EM_PX, SIZE

MAX_CLIPPED_FRACTION = 0.02
_PAD = 32  # render on a padded canvas so we can measure ink that falls outside the box


def _has_all_glyphs(path: str) -> bool:
    font = TTFont(path, lazy=True, fontNumber=0)
    try:
        cmap = font.getBestCmap() or {}
    finally:
        font.close()
    return all(ord(c) in cmap for c in CHARS)


def render_font(path: str) -> np.ndarray | None:
    """Return (52, 64, 64) uint8 bitmaps (0 = paper, 255 = ink), or None.

    None means the file can't be read, lacks one of the 52 letters, or more
    than 2% of its ink falls outside the 64x64 box.
    """
    try:
        if not _has_all_glyphs(path):
            return None
        font = ImageFont.truetype(path, size=EM_PX)
    except Exception:
        return None

    big = SIZE + 2 * _PAD
    out = np.zeros((len(CHARS), SIZE, SIZE), dtype=np.uint8)
    total_ink = 0
    clipped_ink = 0
    for i, c in enumerate(CHARS):
        img = Image.new("L", (big, big), 0)
        try:
            ImageDraw.Draw(img).text(
                (CENTER_COL + _PAD, BASELINE_ROW + _PAD), c, fill=255, font=font, anchor="ms"
            )
        except Exception:  
            return None
        a = np.asarray(img, dtype=np.uint8)
        inside = a[_PAD:_PAD + SIZE, _PAD:_PAD + SIZE]
        total = int(a.sum(dtype=np.int64))
        total_ink += total
        clipped_ink += total - int(inside.sum(dtype=np.int64))
        out[i] = inside

    if total_ink == 0 or clipped_ink / total_ink > MAX_CLIPPED_FRACTION:
        return None
    return out
