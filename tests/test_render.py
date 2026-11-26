from pathlib import Path

import numpy as np

from typefaze.glyphs import CHARS, REF_CHARS, char_id
from typefaze.render import render_font

FIXTURE = str(Path(__file__).parent / "fixtures" / "Lato-Regular.ttf")


def test_chars_and_refs():
    assert len(CHARS) == 52 and char_id("A") == 0 and char_id("z") == 51
    assert REF_CHARS == ["H", "O", "n", "a", "g"]


def test_render_shape_and_dtype():
    g = render_font(FIXTURE)
    assert g.shape == (52, 64, 64) and g.dtype == np.uint8


def test_baseline_convention():
    g = render_font(FIXTURE)
    H = g[char_id("H")]
    assert H[50:, :].max() < 64  # nothing below baseline for H
    assert H[:50, :].max() > 200  # ink above it
    gl = g[char_id("g")]
    assert gl[51:, :].max() > 200  # descender below baseline
    cols = np.where(H.max(0) > 128)[0]
    assert abs((cols.min() + cols.max()) / 2 - 32) <= 3  # centred


def test_font_that_crashes_while_drawing_returns_none(monkeypatch):
    # Real case on Kaggle: FreeType raised "OSError: execution context too long" mid-render.
    from PIL import ImageDraw

    def boom(self, *a, **k):
        raise OSError("execution context too long")

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", boom)
    assert render_font(FIXTURE) is None


def test_missing_glyph_returns_none(tmp_path):
    p = tmp_path / "bad.ttf"
    p.write_bytes(b"nope")
    assert render_font(str(p)) is None


def test_render_respects_max_clipped(monkeypatch):
    import typefaze.render as R

    monkeypatch.setattr(R, "EM_PX", 80)  # oversized letters poke out of the 64x64 box
    assert R.render_font(FIXTURE, max_clipped=0.02) is None
    assert R.render_font(FIXTURE, max_clipped=0.5) is not None
