from pathlib import Path

import numpy as np
from scipy import ndimage

from typefaze.elements import HELDOUT_FAMILIES, PLAIN, TRAIN_FAMILIES, Recipe, apply_recipe, sample_element, sample_recipe
from typefaze.glyphs import CHARS, char_id
from typefaze.render import render_font

FIXTURE = str(Path(__file__).parent / "fixtures" / "Lato-Regular.ttf")
G = render_font(FIXTURE)
REFS = G[[char_id(c) for c in "HOnag"]]
REMOVAL = {"hollow", "stripes", "dashes", "inline"}


def test_family_lists_are_disjoint_and_complete():
    assert len(TRAIN_FAMILIES) == 13 and len(HELDOUT_FAMILIES) == 4
    assert not set(TRAIN_FAMILIES) & set(HELDOUT_FAMILIES)
    assert set(HELDOUT_FAMILIES) == {"end_triangle", "dashes", "attach_heart", "drop_shadow"}


def test_plain_is_identity():
    assert np.array_equal(apply_recipe(REFS, PLAIN), REFS)


def test_every_family_adds_visible_change():
    rng = np.random.default_rng(0)
    for fam in TRAIN_FAMILIES + HELDOUT_FAMILIES:
        out = apply_recipe(REFS, Recipe((sample_element(fam, rng),), None))
        changed = [(np.abs(o.astype(int) - r.astype(int)) > 64).sum() for o, r in zip(out, REFS)]
        # one stroke-end mark adds ~8-12 px, and some letters have a single free end
        need = 8 if fam.startswith("end_") else 20
        assert sum(c >= need for c in changed) >= 4, (fam, changed)
        if fam in REMOVAL:
            assert ((REFS.astype(int) - out.astype(int)) > 64).sum() >= 100, fam
        else:
            added = (out.astype(int) - REFS.astype(int)) > 64
            sizes = []
            for a in added:
                lab, n = ndimage.label(a)
                for sl in ndimage.find_objects(lab):
                    sizes.append(max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start))
            assert max(sizes) >= 3, (fam, max(sizes))


def test_recipe_is_deterministic():
    rec = sample_recipe(np.random.default_rng(1))
    assert np.array_equal(apply_recipe(G, rec), apply_recipe(G, rec))


def test_heldout_never_sampled():
    rng = np.random.default_rng(2)
    for _ in range(5000):
        rec = sample_recipe(rng)
        assert not {name for name, _ in rec.elements} & set(HELDOUT_FAMILIES)


def test_mix_proportions():
    rng = np.random.default_rng(3)
    recs = [sample_recipe(rng) for _ in range(5000)]
    n = np.array([len(r.elements) for r in recs])
    assert abs((n == 0).mean() - 0.4) < 0.03
    assert abs((n == 1).mean() - 0.4) < 0.03
    assert abs((n == 2).mean() - 0.2) < 0.03
    assert abs(np.mean([r.marker_width is not None for r in recs]) - 0.3) < 0.03


def test_same_recipe_all_letters():
    rec = Recipe((sample_element("end_dot", np.random.default_rng(4)),), None)
    out = apply_recipe(G, rec)
    for c in "Hnv":
        i = char_id(c)
        assert ((out[i].astype(int) - G[i].astype(int)) > 64).sum() >= 10, c
    assert out.shape == (len(CHARS), 64, 64) and out.dtype == np.uint8


def test_marker_rendering_changes_stroke_but_keeps_shape():
    out = apply_recipe(REFS, Recipe((), 3.0))
    assert not np.array_equal(out, REFS)
    for o, r in zip(out, REFS):  # most ink stays where the letter is
        ink_o, ink_r = o > 127, r > 127
        assert (ink_o & ink_r).sum() / max(1, ink_r.sum()) > 0.5
