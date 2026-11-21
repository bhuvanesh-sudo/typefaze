import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from typefaze.baseline import pack_bank, retrieve, unpack_bank
from typefaze.glyphs import REF_CHARS, char_id

ROOT = Path(__file__).resolve().parents[1]


def _bank(n=3, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n, 52, 64, 64)) > 0.7).astype(np.uint8) * 255


def test_retrieve_finds_exact_font():
    bank = _bank()
    ids = [char_id(c) for c in REF_CHARS]
    assert retrieve(bank[2][ids], ids, bank) == 2
    assert retrieve(bank[0][ids], ids, bank) == 0


def test_pack_roundtrip_is_1bit():
    bank = _bank(2)
    buf = pack_bank(bank)
    assert len(buf) == 2 * 52 * 512
    assert np.array_equal(unpack_bank(buf, 2), bank)


def test_make_bank_cli(tmp_path):
    from typefaze.data import build_dataset
    import shutil

    ofl = tmp_path / "ofl" / "lato"
    ofl.mkdir(parents=True)
    shutil.copy(ROOT / "tests" / "fixtures" / "Lato-Regular.ttf", ofl / "Lato-Regular.ttf")
    build_dataset(str(tmp_path / "ofl"), str(tmp_path / "d"))
    out = tmp_path / "bank.bin"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_bank.py"), "--data", str(tmp_path / "d"),
                    "--n", "200", "--out", str(out)], check=True)
    assert out.stat().st_size == 52 * 512
    meta = json.loads((tmp_path / "bank.json").read_text())
    assert meta == {"n": 1, "names": ["lato/Lato-Regular.ttf"]}
