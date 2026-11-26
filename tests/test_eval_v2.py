import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from typefaze.data import build_dataset
from typefaze.flow import save_checkpoint
from typefaze.metrics import element_transfer
from typefaze.model import GlyphUNet, StyleEncoder
from typefaze.model_v2 import GlyphUNetV2, RefTokenEncoder

ROOT = Path(__file__).resolve().parents[1]


def test_transfer_metric_sign():
    rng = np.random.default_rng(0)
    plain = (rng.random((5, 64, 64)) > 0.7).astype(np.uint8) * 255
    decorated = plain.copy()
    decorated[:, 5:12, 5:12] = 255
    assert element_transfer(decorated, decorated, plain) > 0
    assert element_transfer(plain, decorated, plain) < 0


def test_eval_v2_table_and_bar(tmp_path):
    for fam in ("lato", "latotwo"):
        d = tmp_path / "ofl" / fam
        d.mkdir(parents=True)
        shutil.copy(ROOT / "tests" / "fixtures" / "Lato-Regular.ttf", d / "Lato-Regular.ttf")
    data = tmp_path / "data"
    build_dataset(str(tmp_path / "ofl"), str(data))
    m = json.loads((data / "manifest.json").read_text())
    m["fonts"][0]["split"], m["fonts"][1]["split"] = "train", "test"
    m["fonts"][1]["category"] = "HANDWRITING"
    (data / "manifest.json").write_text(json.dumps(m))
    save_checkpoint(str(tmp_path / "v1.pt"), GlyphUNet(), StyleEncoder(), step=1, config={"arch": "v1"})
    save_checkpoint(str(tmp_path / "v2.pt"), GlyphUNetV2(), RefTokenEncoder(), step=1, config={"arch": "v2"})
    out = tmp_path / "results.md"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate_v2.py"), "--data", str(data),
                    "--ckpt-v2", str(tmp_path / "v2.pt"), "--ckpt-v1", str(tmp_path / "v1.pt"), "--out", str(out),
                    "--n-fonts", "1", "--steps", "1", "--classifier-epochs", "1", "--families", "end_dot,attach_heart"],
                   check=True)
    text = out.read_text()
    for label in ("Plain", "Handwriting", "end_dot", "attach_heart"):
        assert label in text
    assert sum(line.startswith(("PASS", "FAIL")) for line in text.splitlines()) == 4
    assert (tmp_path / "transfer-gallery.png").exists()
