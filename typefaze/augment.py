"""Make clean font glyphs look a bit more like mouse drawings (applied to reference glyphs only)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _thickness(x: torch.Tensor) -> torch.Tensor:
    """Per sample: thicken (dilate), thin (erode) or leave strokes alone. x: (N,1,H,W), ink = +1."""
    choice = torch.randint(0, 3, (x.shape[0],), device=x.device)
    dil = F.max_pool2d(x, 3, stride=1, padding=1)
    ero = -F.max_pool2d(-x, 3, stride=1, padding=1)
    c = choice.view(-1, 1, 1, 1)
    return torch.where(c == 1, dil, torch.where(c == 2, ero, x))


def _warp(x: torch.Tensor, max_deg: float = 5.0, max_shift_px: float = 2.0, elastic: float = 0.03) -> torch.Tensor:
    """Small rotation, shift and smooth elastic wobble. x: (N,1,H,W), ink = +1."""
    n, _, h, w = x.shape
    ang = (torch.rand(n, device=x.device) * 2 - 1) * math.radians(max_deg)
    shift = (torch.rand(n, 2, device=x.device) * 2 - 1) * (max_shift_px * 2 / w)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, (n, 1, h, w), align_corners=False)
    # smooth random displacement: coarse noise upsampled
    coarse = (torch.rand(n, 2, 4, 4, device=x.device) * 2 - 1) * elastic
    disp = F.interpolate(coarse, size=(h, w), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
    paper0 = (x + 1) / 2  # paper = 0 so out-of-bounds sampling (zeros) is paper
    out = F.grid_sample(paper0, grid + disp, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out * 2 - 1


def augment_refs(refs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """refs: (B,K,1,64,64) in [-1,1]; mask: (B,K) bool. Only masked-in refs are changed."""
    b, k = mask.shape
    flat = refs.reshape(b * k, *refs.shape[2:])
    sel = mask.reshape(-1)
    out = flat.clone()
    if sel.any():
        x = flat[sel]
        x = _warp(_thickness(x))
        out[sel] = x.clamp(-1, 1)
    return out.reshape(refs.shape)
