"""v2 model: reference letters become a set of location-aware tokens, and the generator attends to them.

v1 squeezed the references into one 256-number style vector, which can say "bold, slanted" but not
"there's a dot at the end of every stroke". Here each reference letter becomes an 8x8 grid of tokens
(each token knows which letter it came from and where in it), and the U-Net cross-attends to them at
its 16x16 and 8x8 levels. The global style vector is kept for coarse style via adaptive GroupNorm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import COND_DIM, N_CHARS, STYLE_DIM, Attention, ResBlock, _gn, timestep_embedding

TOKEN_DIM = 128
GRID = 8  # tokens per side, per reference


class RefTokenEncoder(nn.Module):
    def __init__(self, dim: int = STYLE_DIM, token_dim: int = TOKEN_DIM):
        super().__init__()
        chans = [1, 32, 64, 128]
        layers = []
        for a, b in zip(chans[:-1], chans[1:]):  # 64 -> 32 -> 16 -> 8
            layers += [nn.Conv2d(a, b, 3, stride=2, padding=1), _gn(b), nn.SiLU(),
                       nn.Conv2d(b, b, 3, padding=1), _gn(b), nn.SiLU()]
        self.cnn = nn.Sequential(*layers)
        self.to_token = nn.Conv2d(128, token_dim, 1)
        self.pos = nn.Parameter(torch.randn(GRID * GRID, token_dim) * 0.02)
        self.char_tok = nn.Embedding(N_CHARS, token_dim)
        self.char_emb = nn.Embedding(N_CHARS, 128)
        self.mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, refs: torch.Tensor, ref_ids: torch.Tensor, ref_mask: torch.Tensor):
        b, k = ref_ids.shape
        f = self.cnn(refs.reshape(b * k, 1, 64, 64))  # (b*k, 128, 8, 8)
        tok = self.to_token(f).flatten(2).transpose(1, 2).reshape(b, k, GRID * GRID, -1)
        tok = tok + self.pos + self.char_tok(ref_ids)[:, :, None, :]
        tokens = tok.reshape(b, k * GRID * GRID, -1)
        token_mask = ref_mask[:, :, None].expand(b, k, GRID * GRID).reshape(b, k * GRID * GRID)

        per_ref = self.mlp(torch.cat([f.mean(dim=(2, 3)).reshape(b, k, -1), self.char_emb(ref_ids)], dim=-1))
        m = ref_mask.unsqueeze(-1).to(per_ref.dtype)
        style = (per_ref * m).sum(1) / m.sum(1).clamp(min=1.0)
        return style, tokens, token_mask


class CrossAttention(nn.Module):
    """Image features (queries) attend to reference tokens (keys/values). Output starts at zero."""

    def __init__(self, c: int, token_dim: int = TOKEN_DIM, heads: int = 4):
        super().__init__()
        self.norm = _gn(c)
        self.q = nn.Conv2d(c, c, 1)
        self.kv = nn.Linear(token_dim, 2 * c)
        self.out = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.heads = heads

    def forward(self, x: torch.Tensor, tokens: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
        b, c, h, w = x.shape
        hd = c // self.heads
        q = self.q(self.norm(x)).reshape(b, self.heads, hd, h * w).transpose(-1, -2)  # b,H,L,hd
        k, v = self.kv(tokens).chunk(2, dim=-1)  # bt,T,c
        k = k.reshape(k.shape[0], -1, self.heads, hd).transpose(1, 2).expand(b, -1, -1, -1)
        v = v.reshape(v.shape[0], -1, self.heads, hd).transpose(1, 2).expand(b, -1, -1, -1)
        mask = None if token_mask is None else token_mask[:, None, None, :].expand(b, -1, -1, -1)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return x + self.out(a.transpose(-1, -2).reshape(b, c, h, w))


class GlyphUNetV2(nn.Module):
    def __init__(self, style_dim: int = STYLE_DIM, token_dim: int = TOKEN_DIM, n_chars: int = N_CHARS, base: int = 48):
        super().__init__()
        ch = [base, base * 2, base * 3, base * 3]  # 64, 32, 16, 8 (same widths as v1)
        deep = [False, False, True, True]  # self- and cross-attention at 16x16 and 8x8
        self.t_mlp = nn.Sequential(nn.Linear(128, COND_DIM), nn.SiLU(), nn.Linear(COND_DIM, COND_DIM))
        self.char_emb = nn.Embedding(n_chars, COND_DIM)
        self.style_proj = nn.Sequential(nn.Linear(style_dim, COND_DIM), nn.SiLU(), nn.Linear(COND_DIM, COND_DIM))
        self.null_style = nn.Parameter(torch.zeros(style_dim))
        self.null_token = nn.Parameter(torch.zeros(token_dim))

        self.inp = nn.Conv2d(1, ch[0], 3, padding=1)
        self.down, self.down_attn, self.down_cross, self.downsample = (nn.ModuleList() for _ in range(4))
        skips, c = [ch[0]], ch[0]
        for i, co in enumerate(ch):
            for _ in range(2):
                self.down.append(ResBlock(c, co))
                self.down_attn.append(Attention(co) if deep[i] else nn.Identity())
                self.down_cross.append(CrossAttention(co, token_dim) if deep[i] else None)
                c = co
                skips.append(c)
            if i < len(ch) - 1:
                self.downsample.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
                skips.append(c)
        self.mid1, self.mid_attn, self.mid_cross, self.mid2 = ResBlock(c, c), Attention(c), CrossAttention(c, token_dim), ResBlock(c, c)
        self.up, self.up_attn, self.up_cross, self.upsample = (nn.ModuleList() for _ in range(4))
        for i, co in reversed(list(enumerate(ch))):
            for _ in range(3):
                self.up.append(ResBlock(c + skips.pop(), co))
                self.up_attn.append(Attention(co) if deep[i] else nn.Identity())
                self.up_cross.append(CrossAttention(co, token_dim) if deep[i] else None)
                c = co
            if i > 0:
                self.upsample.append(nn.Conv2d(c, c, 3, padding=1))
        self.out = nn.Sequential(_gn(c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))
        self._levels = len(ch)

    def cross_blocks(self) -> list[CrossAttention]:
        return [m for m in self.modules() if isinstance(m, CrossAttention)]

    @staticmethod
    def _cross(block, h, tokens, token_mask):
        return h if block is None else block(h, tokens, token_mask)

    def forward(self, x_t, t, char_id, style, tokens, token_mask=None):
        cond = self.t_mlp(timestep_embedding(t)) + self.char_emb(char_id) + self.style_proj(style)
        h = self.inp(x_t)
        hs, bi = [h], 0
        for lvl in range(self._levels):
            for _ in range(2):
                h = self.down_attn[bi](self.down[bi](h, cond))
                h = self._cross(self.down_cross[bi], h, tokens, token_mask)
                hs.append(h)
                bi += 1
            if lvl < self._levels - 1:
                h = self.downsample[lvl](h)
                hs.append(h)
        h = self.mid_attn(self.mid1(h, cond))
        h = self.mid2(self.mid_cross(h, tokens, token_mask), cond)
        ui = 0
        for lvl in range(self._levels):
            for _ in range(3):
                h = self.up_attn[ui](self.up[ui](torch.cat([h, hs.pop()], dim=1), cond))
                h = self._cross(self.up_cross[ui], h, tokens, token_mask)
                ui += 1
            if lvl < self._levels - 1:
                h = self.upsample[lvl](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.out(h)
