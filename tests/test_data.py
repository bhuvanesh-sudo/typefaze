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


def test_family_category_reads_metadata(tmp_path):
    from typefaze.data import family_category

    (tmp_path / "hw").mkdir()
    (tmp_path / "hw" / "METADATA.pb").write_text('name: "Hw"\ncategory: "HANDWRITING"\n')
    (tmp_path / "none").mkdir()
    assert family_category(tmp_path / "hw") == "HANDWRITING"
    assert family_category(tmp_path / "none") == "UNKNOWN"


def test_handwriting_gets_relaxed_clip(tmp_path, monkeypatch):
    import numpy as np
    import typefaze.data as D

    calls = {}

    def fake_render(path, max_clipped=0.02):
        calls[Path(path).parent.name] = max_clipped
        return np.zeros((52, 64, 64), np.uint8)

    monkeypatch.setattr(D, "render_font", fake_render)
    for fam, cat in (("hw", "HANDWRITING"), ("sans", "SANS_SERIF")):
        (tmp_path / "ofl" / fam).mkdir(parents=True)
        (tmp_path / "ofl" / fam / "A.ttf").write_bytes(b"x")
        (tmp_path / "ofl" / fam / "METADATA.pb").write_text(f'category: "{cat}"\n')
    counts = D.build_dataset(str(tmp_path / "ofl"), str(tmp_path / "out"), handwriting_clip=0.06)
    assert calls == {"hw": 0.06, "sans": 0.02}
    m = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert {f["family"]: f["category"] for f in m["fonts"]} == {"hw": "HANDWRITING", "sans": "SANS_SERIF"}
    assert counts["handwriting_kept"] == 1
