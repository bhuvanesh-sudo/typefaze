import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from typefaze.data import GlyphDatasetV2, build_dataset, handwriting_weights, to_model_space
from typefaze.elements import Recipe, sample_element

FIXTURE = Path(__file__).parent / "fixtures" / "Lato-Regular.ttf"


@pytest.fixture
def built(tmp_path):
    d = tmp_path / "ofl" / "lato"
    d.mkdir(parents=True)
    shutil.copy(FIXTURE, d / "Lato-Regular.ttf")
    build_dataset(str(tmp_path / "ofl"), str(tmp_path / "d"))
    return tmp_path / "d"


def test_refs_and_target_share_recipe(built, monkeypatch):
    import typefaze.data as D

    dot = Recipe((sample_element("end_dot", np.random.default_rng(0)),), None)
    monkeypatch.setattr(D, "sample_recipe", lambda rng, families=None: dot)
    ds = GlyphDatasetV2(str(built / "glyphs.npz"), str(built / "manifest.json"), "train", fixed_ref_prob=1.0, seed=0)
    plain = np.load(built / "glyphs.npz")["glyphs"][0]
    for _ in range(20):
        item = ds[0]
        assert item["recipe"] == "end_dot"
        tid = int(item["target_id"])
        target_plain = to_model_space(plain[tid])
        # letters with stroke ends get dots: the target (never H/O/n/a/g here) must usually differ from plain
        if (item["target"][0] - target_plain).abs().max() > 0.5:
            break
    else:
        pytest.fail("target never carried the element")
    refs = item["refs"][item["ref_mask"]][:, 0]
    plain_refs = to_model_space(plain[item["ref_ids"][item["ref_mask"]].numpy()])
    assert (refs - plain_refs).abs().amax(dim=(1, 2)).gt(0.5).sum() >= 4  # H, n, a, g have stroke ends


def test_item_shapes_match_v1(built):
    ds = GlyphDatasetV2(str(built / "glyphs.npz"), str(built / "manifest.json"), "train", seed=1)
    for _ in range(50):
        item = ds[0]
        assert item["refs"].shape == (8, 1, 64, 64) and item["target"].shape == (1, 64, 64)
        assert int(item["target_id"]) not in item["ref_ids"][item["ref_mask"]].tolist()
        assert item["refs"].min() >= -1 and item["refs"].max() <= 1
        assert isinstance(item["recipe"], str)


def test_handwriting_weighting(tmp_path):
    fonts = [{"file": f"f{i}.ttf", "family": f"f{i}", "split": "train",
              "category": "HANDWRITING" if i < 2 else "SANS_SERIF"} for i in range(10)]
    (tmp_path / "m.json").write_text(json.dumps({"fonts": fonts}))
    w = handwriting_weights(str(tmp_path / "m.json"), "train", factor=4.0)
    assert w.shape == (10,)
    assert torch.isclose(w[:2].sum() / w.sum(), torch.tensor(0.5, dtype=w.dtype))
