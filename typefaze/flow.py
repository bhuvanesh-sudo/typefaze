"""Rectified-flow (flow matching) objective, Euler sampler and checkpoint helpers.

Convention: x0 = Gaussian noise, x1 = real glyph, x_t = (1 - t) * x0 + t * x1.
The model predicts the velocity x1 - x0, i.e. the straight-line direction from noise to glyph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .augment import augment_refs


def encode_style(unet, encoder, batch: dict, p_drop: float = 0.0, augment: bool = True) -> torch.Tensor:
    refs = augment_refs(batch["refs"], batch["ref_mask"]) if augment else batch["refs"]
    style = encoder(refs, batch["ref_ids"], batch["ref_mask"])
    if p_drop > 0:
        drop = torch.rand(style.shape[0], device=style.device) < p_drop
        style = torch.where(drop[:, None], unet.null_style.to(style.dtype).expand_as(style), style)
    return style


def flow_matching_loss(unet, encoder, batch: dict, p_drop: float = 0.1) -> torch.Tensor:
    x1 = batch["target"]
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device)
    tt = t[:, None, None, None]
    x_t = (1 - tt) * x0 + tt * x1
    style = encode_style(unet, encoder, batch, p_drop=p_drop)
    v = unet(x_t, t, batch["target_id"], style)
    return F.mse_loss(v.float(), (x1 - x0).float())


@torch.no_grad()
def sample(unet, style: torch.Tensor, char_id: torch.Tensor, steps: int = 12, cfg: float = 1.0,
           generator: torch.Generator | None = None) -> torch.Tensor:
    """Euler integration from noise (t=0) to glyph (t=1). Returns (B,1,64,64) in [-1, 1]."""
    b = style.shape[0]
    x = torch.randn(b, 1, 64, 64, device=style.device, generator=generator)
    null = unet.null_style.to(style.dtype).expand_as(style)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((b,), i * dt, device=style.device)
        v = unet(x, t, char_id, style)
        if cfg != 1.0:
            v_null = unet(x, t, char_id, null)
            v = v_null + cfg * (v - v_null)
        x = x + dt * v
    return x.clamp(-1, 1)


def save_checkpoint(path: str, unet, encoder, step: int, config: dict, **extra) -> None:
    torch.save({
        "unet": unet.state_dict(),
        "encoder": encoder.state_dict(),
        "null_style": unet.null_style.detach().cpu(),
        "step": step,
        "config": config,
        **extra,
    }, path)


def load_checkpoint(path: str, unet, encoder, map_location: str = "cpu") -> int:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    unet.load_state_dict(ck["unet"])
    encoder.load_state_dict(ck["encoder"])
    return int(ck["step"])
