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
    subprocess.run([sys.executable, str(ROOT / "scripts" / "export_onnx.py"), "--ckpt", str(ck), "--out", str(out), "--fp16",
                    "--batch", "4"], check=True)
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

    B = 4  # the U-Net is exported with a fixed batch size
    x = rng.standard_normal((B, 1, 64, 64)).astype(np.float32)
    t = np.array([0.0, 0.5, 0.9, 1.0], dtype=np.float32)
    cid = np.array([0, 25, 51, 7], dtype=np.int64)
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
    assert meta["batch"] == 4


def test_model_size_budget(tmp_path):
    _, _, out = _export(tmp_path)
    total = sum(p.stat().st_size for p in out.iterdir())
    assert total < 30e6, total


def test_unet_has_static_shapes_and_no_shape_ops(tmp_path):
    # Shape/Gather chains run on the CPU under ONNX Runtime WebGPU and force a GPU->CPU sync each time.
    import onnx
    _, _, out = _export(tmp_path)
    m = onnx.load(str(out / "unet.onnx"))
    dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    assert dims == [4, 1, 64, 64]
    ops = [n.op_type for n in m.graph.node]
    assert ops.count("Shape") == 0
    assert len(ops) < 1000, len(ops)


# ---------------------------------------------------------------- v2

def _export_v2(tmp_path, batch=4):
    from typefaze.model_v2 import GlyphUNetV2, RefTokenEncoder

    torch.manual_seed(0)
    unet, enc = GlyphUNetV2().eval(), RefTokenEncoder().eval()
    with torch.no_grad():
        for blk in unet.cross_blocks():  # make the reference tokens matter, so parity really covers them
            torch.nn.init.normal_(blk.out.weight, std=0.05)
        unet.null_style.normal_()
        unet.null_token.normal_()
    ck = tmp_path / "v2.pt"
    save_checkpoint(str(ck), unet, enc, step=9, config={"arch": "v2"})
    out = tmp_path / "models_v2"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "export_onnx.py"), "--ckpt", str(ck), "--out", str(out), "--fp16",
                    "--batch", str(batch)], check=True)
    return unet, enc, out


def test_v2_onnx_parity_and_files(tmp_path):
    unet, enc, out = _export_v2(tmp_path)
    rng = np.random.default_rng(0)
    refs = rng.uniform(-1, 1, (1, 5, 1, 64, 64)).astype(np.float32)
    ref_ids = np.array([[7, 40, 39, 26, 32]], dtype=np.int64)
    mask = np.ones((1, 5), dtype=bool)
    style, tokens = ort.InferenceSession(str(out / "encoder.onnx")).run(
        ["style", "tokens"], {"refs": refs, "ref_ids": ref_ids, "ref_mask": mask})
    with torch.no_grad():
        s_pt, t_pt, _ = enc(torch.from_numpy(refs), torch.from_numpy(ref_ids), torch.from_numpy(mask))
    assert style.shape == (1, 256) and tokens.shape == (1, 320, 128)
    assert np.abs(style - s_pt.numpy()).max() < 1e-2 and np.abs(tokens - t_pt.numpy()).max() < 1e-2

    B = 4
    x = rng.standard_normal((B, 1, 64, 64)).astype(np.float32)
    t = np.array([0.0, 0.3, 0.6, 0.9], dtype=np.float32)
    cid = np.array([0, 25, 51, 7], dtype=np.int64)
    st = np.repeat(s_pt.numpy(), B, axis=0)
    v = ort.InferenceSession(str(out / "unet.onnx")).run(["v"], {"x_t": x, "t": t, "char_id": cid, "style": st,
                                                                 "tokens": t_pt.numpy()})[0]
    with torch.no_grad():
        v_pt = unet(torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(cid), torch.from_numpy(st), t_pt, None).numpy()
    assert np.abs(v - v_pt).max() < 1e-2

    null_tok = np.fromfile(out / "null_tokens.bin", dtype=np.float32).reshape(320, 128)
    assert np.allclose(null_tok, np.broadcast_to(unet.null_token.detach().numpy(), (320, 128)))
    assert np.allclose(np.fromfile(out / "null_style.bin", dtype=np.float32), unet.null_style.detach().numpy())
    meta = json.loads((out / "meta.json").read_text())
    assert meta["arch"] == "v2" and meta["batch"] == 4 and meta["version"] == 9


def test_v2_no_shape_ops_and_size(tmp_path):
    import onnx

    _, _, out = _export_v2(tmp_path, batch=52)
    m = onnx.load(str(out / "unet.onnx"))
    assert [n.op_type for n in m.graph.node].count("Shape") == 0
    assert sum(p.stat().st_size for p in out.iterdir()) < 50e6
