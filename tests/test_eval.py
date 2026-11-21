import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from typefaze.classifier import GlyphClassifier
from typefaze.data import build_dataset
from typefaze.flow import save_checkpoint
from typefaze.metrics import family_base, l1, ssim
from typefaze.model import GlyphUNet, StyleEncoder

ROOT = Path(__file__).resolve().parents[1]


def test_metrics_identity_and_difference():
    rng = np.random.default_rng(0)
    x = (rng.random((64, 64)) * 255).astype(np.uint8)
    assert ssim(x, x) == 1.0
    assert l1(x, x) == 0.0
    y = 255 - x
    assert ssim(x, y) < 0.5
    assert 0 < l1(x, y) <= 1


def test_family_base_groups_siblings():
    assert family_base("encodesanscondensed") == family_base("encodesans") == "encodesans"
    assert family_base("firasansextracondensed") == "firasans"
    assert family_base("notoserifdisplay") == "notoserif"
    assert family_base("lato") == "lato"


def test_classifier_output_shape():
    assert GlyphClassifier()(torch.zeros(3, 1, 64, 64)).shape == (3, 52)


def test_eval_table_rows(tmp_path):
    # two "families": lato (train) and a copy (test), so every split is non-empty
    for fam in ("lato", "latocopy"):
        d = tmp_path / "ofl" / fam
        d.mkdir(parents=True)
        shutil.copy(ROOT / "tests" / "fixtures" / "Lato-Regular.ttf", d / "Lato-Regular.ttf")
    data = tmp_path / "data"
    build_dataset(str(tmp_path / "ofl"), str(data))
    import json

    m = json.loads((data / "manifest.json").read_text())
    m["fonts"][0]["split"], m["fonts"][1]["split"] = "train", "test"
    (data / "manifest.json").write_text(json.dumps(m))

    save_checkpoint(str(tmp_path / "ck.pt"), GlyphUNet(), StyleEncoder(), step=1, config={})
    out = tmp_path / "results.md"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--data", str(data), "--ckpt", str(tmp_path / "ck.pt"),
                    "--out", str(out), "--n-fonts", "2", "--steps", "2", "--classifier-epochs", "1", "--cfg", "2.0"], check=True)
    text = out.read_text()
    for label in ("Real (ceiling)", "TypeFaze", "Nearest-font baseline", "Classifier accuracy", "Mean L1", "Mean SSIM"):
        assert label in text
    assert (tmp_path / "failures.png").exists()
    assert "guidance 2.0" in text


def test_classifier_training_converts_in_batches(tmp_path, monkeypatch):
    # Regression: converting every training glyph to float32 at once ran a 3,194-font run out of memory.
    import typefaze.classifier as C

    for k in range(12):  # 12 fonts x 52 letters = 624 glyphs, more than one batch's worth
        d = tmp_path / "ofl" / f"lato{k}"
        d.mkdir(parents=True)
        shutil.copy(ROOT / "tests" / "fixtures" / "Lato-Regular.ttf", d / "Lato-Regular.ttf")
    build_dataset(str(tmp_path / "ofl"), str(tmp_path / "data"))
    import json
    mp = tmp_path / "data" / "manifest.json"
    m = json.loads(mp.read_text())
    for f in m["fonts"]:
        f["split"] = "train"
    mp.write_text(json.dumps(m))
    real = C.to_model_space

    def guarded(a):
        assert len(a) <= 512, f"converted {len(a)} glyphs at once"
        return real(a)

    monkeypatch.setattr(C, "to_model_space", guarded)
    C.train_classifier(str(tmp_path / "data" / "glyphs.npz"), str(tmp_path / "data" / "manifest.json"), epochs=1, batch=16)
