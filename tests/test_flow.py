import shutil
from pathlib import Path

import pytest
import torch

from typefaze.data import GlyphDataset, build_dataset
from typefaze.flow import flow_matching_loss, load_checkpoint, sample, save_checkpoint
from typefaze.model import GlyphUNet, StyleEncoder

FIXTURE = Path(__file__).parent / "fixtures" / "Lato-Regular.ttf"


def _batch(B=4):
    mask = torch.zeros(B, 8, dtype=torch.bool)
    mask[:, :5] = True
    return {
        "refs": torch.rand(B, 8, 1, 64, 64) * 2 - 1,
        "ref_ids": torch.randint(0, 52, (B, 8)),
        "ref_mask": mask,
        "target": torch.rand(B, 1, 64, 64) * 2 - 1,
        "target_id": torch.randint(0, 52, (B,)),
    }


def test_loss_is_scalar_and_finite():
    loss = flow_matching_loss(GlyphUNet(), StyleEncoder(), _batch())
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()


def test_sample_shape_and_range():
    unet = GlyphUNet().eval()
    out = sample(unet, torch.randn(3, 256), torch.tensor([0, 7, 51]), steps=4, cfg=1.5)
    assert out.shape == (3, 1, 64, 64)
    assert out.min() >= -1 and out.max() <= 1


def test_resume_restores_step(tmp_path):
    unet, enc = GlyphUNet(), StyleEncoder()
    save_checkpoint(str(tmp_path / "c.pt"), unet, enc, step=123, config={"lr": 1e-4})
    ck = torch.load(tmp_path / "c.pt", weights_only=False)
    assert set(["unet", "encoder", "null_style", "step", "config"]) <= set(ck)
    unet2, enc2 = GlyphUNet(), StyleEncoder()
    step = load_checkpoint(str(tmp_path / "c.pt"), unet2, enc2)
    assert step == 123
    assert torch.equal(unet2.null_style, unet.null_style)


@pytest.mark.slow
def test_overfit_one_font(tmp_path):
    torch.manual_seed(0)
    ofl = tmp_path / "ofl" / "lato"
    ofl.mkdir(parents=True)
    shutil.copy(FIXTURE, ofl / "Lato-Regular.ttf")
    build_dataset(str(tmp_path / "ofl"), str(tmp_path / "d"))
    ds = GlyphDataset(str(tmp_path / "d" / "glyphs.npz"), str(tmp_path / "d" / "manifest.json"), "train")
    unet, enc = GlyphUNet(), StyleEncoder()
    opt = torch.optim.AdamW(list(unet.parameters()) + list(enc.parameters()), lr=5e-4)
    losses = []
    for _ in range(300):
        items = [ds[0] for _ in range(8)]
        batch = {k: torch.stack([it[k] for it in items]) for k in items[0]}
        loss = flow_matching_loss(unet, enc, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    first = sum(losses[:5]) / 5
    last = sum(losses[-20:]) / 20
    assert last < 0.3 * first, (first, last)


# ---------------------------------------------------------------- v2

def _v2():
    from typefaze.model_v2 import GlyphUNetV2, RefTokenEncoder

    return GlyphUNetV2(), RefTokenEncoder()


def test_v2_loss_finite_and_backward():
    unet, enc = _v2()
    loss = flow_matching_loss(unet, enc, _batch(), p_drop=0.5)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert unet.null_token.grad is not None  # dropped examples train the null token


def test_v2_sample_shape():
    from typefaze.flow import encode

    unet, enc = _v2()
    unet.eval(), enc.eval()
    b = _batch(1)
    with torch.no_grad():
        cond = encode(unet, enc, b, augment=False)
        assert set(cond) == {"style", "tokens", "token_mask"}
        cond = {"style": cond["style"].expand(3, -1), "tokens": cond["tokens"], "token_mask": cond["token_mask"]}
        out = sample(unet, cond, torch.tensor([0, 7, 51]), steps=3, cfg=2.0)
    assert out.shape == (3, 1, 64, 64) and out.min() >= -1 and out.max() <= 1


@pytest.mark.slow
def test_v2_overfit(tmp_path):
    import numpy as np

    import typefaze.data as D
    from typefaze.elements import Recipe, sample_element

    torch.manual_seed(0)
    ofl = tmp_path / "ofl" / "lato"
    ofl.mkdir(parents=True)
    shutil.copy(FIXTURE, ofl / "Lato-Regular.ttf")
    build_dataset(str(tmp_path / "ofl"), str(tmp_path / "d"))
    dot = Recipe((sample_element("end_dot", np.random.default_rng(0)),), None)
    D.sample_recipe = lambda rng, families=None: dot
    ds = D.GlyphDatasetV2(str(tmp_path / "d" / "glyphs.npz"), str(tmp_path / "d" / "manifest.json"), "train", seed=0)
    unet, enc = _v2()
    opt = torch.optim.AdamW(list(unet.parameters()) + list(enc.parameters()), lr=5e-4)
    losses = []
    for _ in range(300):
        items = [ds[0] for _ in range(8)]
        batch = {k: torch.stack([it[k] for it in items]) for k in items[0] if k != "recipe"}
        loss = flow_matching_loss(unet, enc, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-20:]) / 20 < 0.3 * sum(losses[:5]) / 5


def test_train_cli_v2_smoke(tmp_path):
    import subprocess
    import sys

    ofl = tmp_path / "ofl" / "lato"
    ofl.mkdir(parents=True)
    shutil.copy(FIXTURE, ofl / "Lato-Regular.ttf")
    build_dataset(str(tmp_path / "ofl"), str(tmp_path / "d"))
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts" / "train.py"), "--arch", "v2", "--data", str(tmp_path / "d"),
                    "--out", str(tmp_path / "run"), "--batch", "2", "--max-steps", "3", "--workers", "0",
                    "--grid-steps", "2"], check=True)
    ck = torch.load(tmp_path / "run" / "ckpt_latest.pt", weights_only=False)
    assert ck["config"]["arch"] == "v2" and ck["step"] == 3
    assert any(p.name.startswith("samples_") for p in (tmp_path / "run").iterdir())
