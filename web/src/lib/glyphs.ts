// Shared constants. Must match typefaze/glyphs.py exactly.

export const CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
export const REF_CHARS = ["H", "O", "n", "a", "g"] as const;

export const SIZE = 64; // glyph bitmap side
export const SCALE = 4; // drawing canvas = 256 px = 4 x 64
export const EM_PX = 48;
export const BASELINE_ROW = 50;
export const CENTER_COL = 32;
export const GUIDE_ROWS = { cap: 16, xheight: 26, baseline: 50, descender: 62 } as const;

/** 64x64 = 4096 values in [-1, 1], ink = +1, paper = -1, row-major from the top. */
export type GlyphBitmap = Float32Array;
export type RefChar = (typeof REF_CHARS)[number];
export type RefSet = Record<RefChar, GlyphBitmap>;

export const charId = (c: string): number => {
  const i = CHARS.indexOf(c);
  if (i < 0 || c.length !== 1) throw new Error(`not a TypeFaze character: ${c}`);
  return i;
};
