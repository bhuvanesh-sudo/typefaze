import torch

from typefaze.model_v2 import GlyphUNetV2, RefTokenEncoder


def _inputs(B=3, K=8, n_valid=5):
    refs = torch.rand(B, K, 1, 64, 64) * 2 - 1
    ids = torch.randint(0, 52, (B, K))
    mask = torch.zeros(B, K, dtype=torch.bool)
    mask[:, :n_valid] = True
    return refs, ids, mask


def test_shapes_v2():
    enc, unet = RefTokenEncoder().eval(), GlyphUNetV2().eval()
    refs, ids, mask = _inputs()
    style, tokens, tmask = enc(refs, ids, mask)
    assert style.shape == (3, 256) and tokens.shape == (3, 8 * 64, 128) and tmask.shape == (3, 512)
    assert tmask[:, : 5 * 64].all() and not tmask[:, 5 * 64:].any()
    v = unet(torch.randn(3, 1, 64, 64), torch.rand(3), torch.randint(0, 52, (3,)), style, tokens, tmask)
    assert v.shape == (3, 1, 64, 64)
    assert unet.null_style.shape == (256,) and unet.null_token.shape == (128,)


def test_param_budget_v2():
    n = sum(p.numel() for m in (RefTokenEncoder(), GlyphUNetV2()) for p in m.parameters())
    assert n <= 25e6, n


def _nonzero_cross_attention(unet):
    # cross-attention output projections start at zero (a stable start for training); give them weights
    torch.manual_seed(0)
    for blk in unet.cross_blocks():
        torch.nn.init.normal_(blk.out.weight, std=0.05)
    return unet


def test_token_broadcast():
    unet = _nonzero_cross_attention(GlyphUNetV2().eval())
    B, T = 4, 320
    x, t, c, s = torch.randn(B, 1, 64, 64), torch.rand(B), torch.randint(0, 52, (B,)), torch.randn(B, 256)
    tok = torch.randn(1, T, 128)
    with torch.no_grad():
        a = unet(x, t, c, s, tok, None)
        b = unet(x, t, c, s, tok.expand(B, -1, -1), torch.ones(B, T, dtype=torch.bool))
    assert torch.allclose(a, b, atol=1e-5)


def test_masked_tokens_ignored():
    unet = _nonzero_cross_attention(GlyphUNetV2().eval())
    B, T = 2, 320
    x, t, c, s = torch.randn(B, 1, 64, 64), torch.rand(B), torch.randint(0, 52, (B,)), torch.randn(B, 256)
    tok = torch.randn(B, T, 128)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, 200:] = False
    tok2 = tok.clone()
    tok2[:, 200:] = torch.randn(B, T - 200, 128)
    with torch.no_grad():
        assert torch.allclose(unet(x, t, c, s, tok, mask), unet(x, t, c, s, tok2, mask), atol=1e-5)
        tok3 = tok.clone()
        tok3[:, 10] += 5.0  # a *valid* token does matter
        assert not torch.allclose(unet(x, t, c, s, tok, mask), unet(x, t, c, s, tok3, mask), atol=1e-4)


def test_tokens_are_local():
    # a mark in one corner of a reference changes the tokens near that corner far more than elsewhere
    enc = RefTokenEncoder().eval()
    refs = torch.full((1, 1, 1, 64, 64), -1.0)
    ids = torch.zeros(1, 1, dtype=torch.long)
    mask = torch.ones(1, 1, dtype=torch.bool)
    marked = refs.clone()
    marked[..., 2:10, 2:10] = 1.0  # top-left
    with torch.no_grad():
        _, a, _ = enc(refs, ids, mask)
        _, b, _ = enc(marked, ids, mask)
    diff = (a - b).norm(dim=-1).reshape(8, 8)
    assert diff[:3, :3].mean() > 3 * diff[5:, 5:].mean()
