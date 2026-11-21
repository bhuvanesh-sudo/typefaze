import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from typefaze.flow import save_checkpoint
from typefaze.model import GlyphUNet, StyleEncoder

ROOT = Path(__file__).resolve().parents[1]


def _export(tmp_path):
    torch.manual_seed(0)
    unet, enc = GlyphUNet().eval(), StyleEncoder().eval()
    with torch.no_grad():
        unet.null_style.normal_()
    ck = tmp_path / "ck.pt"
    save_checkpoint(str(ck), unet, enc, step=7, config={})
    out = tmp_path / "models"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "export_onnx.py"), "--ckpt", str(ck), "--out", str(out), "--fp16"],
                   check=True)
    return unet, enc, out


def test_onnx_parity(tmp_path):
    unet, enc, out = _export(tmp_path)
    rng = np.random.default_rng(0)
    refs = rng.uniform(-1, 1, (1, 5, 1, 64, 64)).astype(np.float32)
    ref_ids = np.array([[7, 40, 39, 26, 32]], dtype=np.int64)
    mask = np.ones((1, 5), dtype=bool)
    s_enc = ort.InferenceSession(str(out / "encoder.onnx"))
    style = s_enc.run(["style"], {"refs": refs, "ref_ids": ref_ids, "ref_mask": mask})[0]
    with torch.no_grad():
        style_pt = enc(torch.from_numpy(refs), torch.from_numpy(ref_ids), torch.from_numpy(mask)).numpy()
    assert style.shape == (1, 256)
    assert np.abs(style - style_pt).max() < 1e-2

    B = 3  # dynamic batch
    x = rng.standard_normal((B, 1, 64, 64)).astype(np.float32)
    t = np.array([0.0, 0.5, 0.9], dtype=np.float32)
    cid = np.array([0, 25, 51], dtype=np.int64)
    st = np.repeat(style_pt, B, axis=0).astype(np.float32)
    s_unet = ort.InferenceSession(str(out / "unet.onnx"))
    v = s_unet.run(["v"], {"x_t": x, "t": t, "char_id": cid, "style": st})[0]
    with torch.no_grad():
        v_pt = unet(torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(cid), torch.from_numpy(st)).numpy()
    assert v.shape == (B, 1, 64, 64)
    assert np.abs(v - v_pt).max() < 1e-2

    null = np.fromfile(out / "null_style.bin", dtype=np.float32)
    assert null.shape == (256,) and np.allclose(null, unet.null_style.detach().numpy())
    meta = json.loads((out / "meta.json").read_text())
    assert meta["steps"] == 12 and meta["wasmSteps"] == 4 and meta["cfg"] == 1.0 and meta["version"] == 7


def test_model_size_budget(tmp_path):
    _, _, out = _export(tmp_path)
    total = sum(p.stat().st_size for p in out.iterdir())
    assert total < 30e6, total
