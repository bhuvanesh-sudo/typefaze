"""Style encoder + conditional U-Net for flow matching on 64x64 glyphs."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CHARS = 52
STYLE_DIM = 256
COND_DIM = 512


def _gn(c: int) -> nn.GroupNorm:
    groups = max(g for g in range(1, 33) if c % g == 0)
    return nn.GroupNorm(groups, c)


class StyleEncoder(nn.Module):
    """Encode k reference glyphs (each tagged with which letter it is) into one style vector."""

    def __init__(self, dim: int = STYLE_DIM):
        super().__init__()
        chans = [1, 32, 64, 128, 256]
        layers = []
        for a, b in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(a, b, 3, stride=2, padding=1), _gn(b), nn.SiLU(),
                       nn.Conv2d(b, b, 3, padding=1), _gn(b), nn.SiLU()]
        self.cnn = nn.Sequential(*layers)  # 64 -> 4
        self.char_emb = nn.Embedding(N_CHARS, 256)
        self.mlp = nn.Sequential(nn.Linear(512, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, refs: torch.Tensor, ref_ids: torch.Tensor, ref_mask: torch.Tensor) -> torch.Tensor:
        b, k = ref_ids.shape
        feats = self.cnn(refs.reshape(b * k, 1, 64, 64)).mean(dim=(2, 3)).reshape(b, k, -1)
        per_ref = self.mlp(torch.cat([feats, self.char_emb(ref_ids)], dim=-1))
        m = ref_mask.unsqueeze(-1).to(per_ref.dtype)
        return (per_ref * m).sum(1) / m.sum(1).clamp(min=1.0)


def timestep_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    """Residual block with adaptive GroupNorm: the condition sets each norm's scale and shift."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.n1, self.c1 = _gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = _gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.cond = nn.Linear(COND_DIM, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.cond(cond)[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(F.silu(h))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.norm = _gn(c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.out = nn.Conv2d(c, c, 1)
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        a = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.out(a.transpose(-1, -2).reshape(b, c, h, w))


class GlyphUNet(nn.Module):
    """Predicts the flow-matching velocity for one target letter, given time, letter id and style."""

    def __init__(self, style_dim: int = STYLE_DIM, n_chars: int = N_CHARS, base: int = 48):
        super().__init__()
        ch = [base, base * 2, base * 3, base * 3]  # 64, 32, 16, 8
        attn = [False, False, True, True]
        self.t_mlp = nn.Sequential(nn.Linear(128, COND_DIM), nn.SiLU(), nn.Linear(COND_DIM, COND_DIM))
        self.char_emb = nn.Embedding(n_chars, COND_DIM)
        self.style_proj = nn.Sequential(nn.Linear(style_dim, COND_DIM), nn.SiLU(), nn.Linear(COND_DIM, COND_DIM))
        self.null_style = nn.Parameter(torch.zeros(style_dim))

        self.inp = nn.Conv2d(1, ch[0], 3, padding=1)
        self.down = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsample = nn.ModuleList()
        skips = [ch[0]]
        c = ch[0]
        for i, co in enumerate(ch):
            for _ in range(2):
                self.down.append(ResBlock(c, co))
                self.down_attn.append(Attention(co) if attn[i] else nn.Identity())
                c = co
                skips.append(c)
            if i < len(ch) - 1:
                self.downsample.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
                skips.append(c)
        self.mid1, self.mid_attn, self.mid2 = ResBlock(c, c), Attention(c), ResBlock(c, c)

        self.up = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i, co in reversed(list(enumerate(ch))):
            for _ in range(3):
                self.up.append(ResBlock(c + skips.pop(), co))
                self.up_attn.append(Attention(co) if attn[i] else nn.Identity())
                c = co
            if i > 0:
                self.upsample.append(nn.Conv2d(c, c, 3, padding=1))
        self.out = nn.Sequential(_gn(c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))
        self._levels = len(ch)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, char_id: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        cond = self.t_mlp(timestep_embedding(t)) + self.char_emb(char_id) + self.style_proj(style)
        h = self.inp(x_t)
        hs = [h]
        bi = 0
        for lvl in range(self._levels):
            for _ in range(2):
                h = self.down_attn[bi](self.down[bi](h, cond))
                hs.append(h)
                bi += 1
            if lvl < self._levels - 1:
                h = self.downsample[lvl](h)
                hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, cond)), cond)
        ui = 0
        for lvl in range(self._levels):
            for _ in range(3):
                h = self.up_attn[ui](self.up[ui](torch.cat([h, hs.pop()], dim=1), cond))
                ui += 1
            if lvl < self._levels - 1:
                h = self.upsample[lvl](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.out(h)
