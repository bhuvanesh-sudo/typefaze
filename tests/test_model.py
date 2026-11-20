import torch

from typefaze.augment import augment_refs
from typefaze.model import GlyphUNet, StyleEncoder


def _inputs(B=4, K=8):
    refs = torch.rand(B, K, 1, 64, 64) * 2 - 1
    ids = torch.randint(0, 52, (B, K))
    mask = torch.zeros(B, K, dtype=torch.bool)
    mask[:, :5] = True
    return refs, ids, mask


def test_shapes():
    enc, unet = StyleEncoder(), GlyphUNet()
    refs, ids, mask = _inputs()
    style = enc(refs, ids, mask)
    assert style.shape == (4, 256)
    v = unet(torch.randn(4, 1, 64, 64), torch.rand(4), torch.randint(0, 52, (4,)), style)
    assert v.shape == (4, 1, 64, 64)
    assert unet.null_style.shape == (256,) and unet.null_style.requires_grad


def test_param_budget():
    n = sum(p.numel() for m in (StyleEncoder(), GlyphUNet()) for p in m.parameters() if p.requires_grad)
    assert 10e6 <= n <= 25e6, n


def test_mask_ignored():
    enc = StyleEncoder().eval()
    refs, ids, mask = _inputs()
    a = enc(refs, ids, mask)
    refs2 = refs.clone()
    refs2[:, 5:] = torch.rand_like(refs2[:, 5:])
    ids2 = ids.clone()
    ids2[:, 5:] = 0
    b = enc(refs2, ids2, mask)
    assert torch.allclose(a, b, atol=1e-5)


def test_augment_preserves_masked_out():
    torch.manual_seed(0)
    refs, _, mask = _inputs()
    out = augment_refs(refs, mask)
    assert out.shape == refs.shape
    assert torch.equal(out[:, 5:], refs[:, 5:])
    assert not torch.equal(out[:, :5], refs[:, :5])
    assert out.min() >= -1 and out.max() <= 1
