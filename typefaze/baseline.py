"""Nearest-existing-font baseline: pick the bank font whose reference letters look most like yours."""

from __future__ import annotations

import numpy as np

GLYPH_BYTES = 64 * 64 // 8  # 1-bit packed


def retrieve(refs: np.ndarray, ref_ids: list[int], bank: np.ndarray) -> int:
    """refs (k,64,64) uint8, bank (N,52,64,64) uint8 -> index of the font with the lowest mean L1 on ref letters."""
    q = refs.astype(np.float32)
    cand = bank[:, ref_ids].astype(np.float32)
    dist = np.abs(cand - q[None]).mean(axis=(1, 2, 3))
    return int(np.argmin(dist))


def pack_bank(bank: np.ndarray) -> bytes:
    """(N,52,64,64) uint8 -> 1 bit per pixel (ink if >= 128), row-major, MSB first."""
    return np.packbits(bank >= 128, axis=-1).tobytes()


def unpack_bank(buf: bytes, n: int) -> np.ndarray:
    bits = np.frombuffer(buf, dtype=np.uint8).reshape(n, 52, 64, 8)
    return np.unpackbits(bits, axis=-1).astype(np.uint8) * 255
