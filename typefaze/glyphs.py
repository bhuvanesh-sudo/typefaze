"""Character set shared by every part of TypeFaze (mirrored in web/src/lib/glyphs.ts)."""

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
REF_CHARS = ["H", "O", "n", "a", "g"]

SIZE = 64          # bitmap side in pixels
EM_PX = 48         # em size in pixels
BASELINE_ROW = 50  # baseline row, 0-indexed from the top
CENTER_COL = 32    # glyphs are centred on their advance width here


def char_id(c: str) -> int:
    """Index of a character in CHARS; raises ValueError for anything else."""
    i = CHARS.find(c)
    if i < 0 or len(c) != 1:
        raise ValueError(f"not a TypeFaze character: {c!r}")
    return i
