"""Rectified-flow (flow matching) objective, Euler sampler and checkpoint helpers.

Convention: x0 = Gaussian noise, x1 = real glyph, x_t = (1 - t) * x0 + t * x1.
The model predicts the velocity x1 - x0, i.e. the straight-line direction from noise to glyph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .augment import augment_refs


def encode(unet, encoder, batch: dict, p_drop: float = 0.0, augment: bool = True) -> dict:
    """Turn the reference letters into conditioning for the U-Net.

    v1 returns {"style"}; v2 also returns {"tokens", "token_mask"}. With probability p_drop an example's
    references are replaced by the learned "no references" values, which is what guidance compares against.
    """
    refs = augment_refs(batch["refs"], batch["ref_mask"]) if augment else batch["refs"]
    out = encoder(refs, batch["ref_ids"], batch["ref_mask"])
    if isinstance(out, tuple):
        style, tokens, token_mask = out
        cond = {"style": style, "tokens": tokens, "token_mask": token_mask}
    else:
        cond = {"style": out}
    if p_drop > 0:
        drop = torch.rand(cond["style"].shape[0], device=cond["style"].device) < p_drop
        cond = _null_where(unet, cond, drop)
    return cond


def encode_style(unet, encoder, batch: dict, p_drop: float = 0.0, augment: bool = True) -> torch.Tensor:
    """v1 compatibility: just the style vector."""
    return encode(unet, encoder, batch, p_drop, augment)["style"]


def _null_where(unet, cond: dict, drop: torch.Tensor) -> dict:
    style = cond["style"]
    out = {"style": torch.where(drop[:, None], unet.null_style.to(style.dtype).expand_as(style), style)}
    if "tokens" in cond:
        tok = cond["tokens"]
        null_tok = unet.null_token.to(tok.dtype).expand_as(tok)
        out["tokens"] = torch.where(drop[:, None, None], null_tok, tok)
        out["token_mask"] = torch.where(drop[:, None], torch.ones_like(cond["token_mask"]), cond["token_mask"])
    return out


def _null_cond(unet, cond: dict) -> dict:
    style = cond["style"]
    out = {"style": unet.null_style.to(style.dtype).expand_as(style)}
    if "tokens" in cond:
        out["tokens"] = unet.null_token.to(cond["tokens"].dtype).expand_as(cond["tokens"])
        out["token_mask"] = torch.ones_like(cond["token_mask"]) if cond["token_mask"] is not None else None
    return out


def velocity(unet, x_t, t, char_id, cond: dict) -> torch.Tensor:
    if "tokens" in cond:
        return unet(x_t, t, char_id, cond["style"], cond["tokens"], cond["token_mask"])
    return unet(x_t, t, char_id, cond["style"])


def flow_matching_loss(unet, encoder, batch: dict, p_drop: float = 0.1) -> torch.Tensor:
    x1 = batch["target"]
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device)
    tt = t[:, None, None, None]
    x_t = (1 - tt) * x0 + tt * x1
    cond = encode(unet, encoder, batch, p_drop=p_drop)
    v = velocity(unet, x_t, t, batch["target_id"], cond)
    return F.mse_loss(v.float(), (x1 - x0).float())


@torch.no_grad()
def sample(unet, cond, char_id: torch.Tensor, steps: int = 12, cfg: float = 1.0,
           generator: torch.Generator | None = None) -> torch.Tensor:
    """Euler integration from noise (t=0) to glyph (t=1). Returns (B,1,64,64) in [-1, 1].

    `cond` is a dict from encode() (style batch B; tokens batch B or 1), or a v1 style tensor.
    """
    if isinstance(cond, torch.Tensor):
        cond = {"style": cond}
    style = cond["style"]
    b = style.shape[0]
    x = torch.randn(b, 1, 64, 64, device=style.device, generator=generator)
    null = _null_cond(unet, cond) if cfg != 1.0 else None
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((b,), i * dt, device=style.device)
        v = velocity(unet, x, t, char_id, cond)
        if null is not None:
            v_null = velocity(unet, x, t, char_id, null)
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
