import json
import shutil
from pathlib import Path

import pytest
import torch

from typefaze.data import GlyphDataset, build_dataset, split_families
from typefaze.glyphs import REF_CHARS, char_id

FIXTURE = Path(__file__).parent / "fixtures" / "Lato-Regular.ttf"


def test_split_is_by_family_and_deterministic():
    fams = [f"fam{i:03d}" for i in range(100)]
    a = split_families(fams, seed=0)
    b = split_families(list(reversed(fams)), seed=0)
    assert [len(a[k]) for k in ("train", "val", "test")] == [90, 5, 5]
    assert set(a["train"]) | set(a["val"]) | set(a["test"]) == set(fams)
    assert not (set(a["train"]) & set(a["val"]) or set(a["train"]) & set(a["test"]) or set(a["val"]) & set(a["test"]))
    assert a == b


@pytest.fixture
def built(tmp_path):
    ofl = tmp_path / "ofl" / "lato"
    ofl.mkdir(parents=True)
    shutil.copy(FIXTURE, ofl / "Lato-Regular.ttf")
    (tmp_path / "ofl" / "broken").mkdir()
    (tmp_path / "ofl" / "broken" / "Broken.ttf").write_bytes(b"nope")
    out = tmp_path / "out"
    counts = build_dataset(str(tmp_path / "ofl"), str(out))
    return out, counts


def test_build_dataset_on_fixture_dir(built):
    out, counts = built
    import numpy as np

    z = np.load(out / "glyphs.npz")
    assert z["glyphs"].shape == (1, 52, 64, 64)
    m = json.loads((out / "manifest.json").read_text())
    assert m["fonts"][0]["family"] == "lato"
    assert counts["kept"] == 1 and counts["skipped"] == 1


def test_dataset_item_shapes_and_no_target_in_refs(built):
    out, _ = built
    ds = GlyphDataset(str(out / "glyphs.npz"), str(out / "manifest.json"), "train")
    ref_ids_fixed = torch.tensor([char_id(c) for c in REF_CHARS])
    saw_fixed = False
    for i in range(200):
        item = ds[0]
        assert item["refs"].shape == (8, 1, 64, 64)
        assert item["ref_ids"].shape == (8,) and item["ref_mask"].shape == (8,)
        assert item["target"].shape == (1, 64, 64)
        active = item["ref_ids"][item["ref_mask"]]
        assert int(item["target_id"]) not in active.tolist()
        n = int(item["ref_mask"].sum())
        assert 3 <= n <= 8
        if n == 5 and torch.equal(active, ref_ids_fixed):
            saw_fixed = True
        for t in (item["refs"], item["target"]):
            assert t.min() >= -1 and t.max() <= 1
    assert saw_fixed
