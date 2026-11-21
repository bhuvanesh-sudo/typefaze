// A Generator turns the 5 reference letters into all 52 letters (CHARS order).
// Two implementations share this interface: the nearest-font baseline (here) and the ONNX model.

import { CHARS, REF_CHARS, SIZE, charId, type GlyphBitmap, type RefSet } from "./glyphs";

/** Returns glyphs for `chars` (indices into CHARS, default all 52), in the same order. */
export type Generator = (refs: RefSet, chars?: number[]) => Promise<GlyphBitmap[]>;

export const ALL_CHARS = Array.from(CHARS, (_, i) => i);

const GLYPH_BYTES = (SIZE * SIZE) / 8;

/** Decode one 1-bit packed glyph (MSB first) from a bank into [-1, 1]. */
export function unpackGlyph(buf: ArrayBuffer, font: number, char: number): GlyphBitmap {
  const bytes = new Uint8Array(buf, (font * CHARS.length + char) * GLYPH_BYTES, GLYPH_BYTES);
  const g = new Float32Array(SIZE * SIZE);
  for (let i = 0; i < g.length; i++) g[i] = (bytes[i >> 3] >> (7 - (i & 7))) & 1 ? 1 : -1;
  return g;
}

type BankLoader = () => Promise<{ buf: ArrayBuffer; n: number }>;

export function createBaselineGenerator(load: BankLoader): Generator {
  let bank: Promise<{ buf: ArrayBuffer; n: number }> | null = null;
  return async (refs, chars = ALL_CHARS) => {
    bank ??= load();
    const { buf, n } = await bank;
    let best = 0;
    let bestDist = Infinity;
    for (let f = 0; f < n; f++) {
      let d = 0;
      for (const c of REF_CHARS) {
        const cand = unpackGlyph(buf, f, charId(c));
        const q = refs[c];
        for (let i = 0; i < cand.length; i++) d += Math.abs(cand[i] - q[i]);
      }
      if (d < bestDist) {
        bestDist = d;
        best = f;
      }
    }
    return chars.map((i) => unpackGlyph(buf, best, i));
  };
}

/** Nearest existing font from the bundled bank of ~200 Google Fonts. */
export const baselineGenerator: Generator = createBaselineGenerator(async () => {
  const base = import.meta.env.BASE_URL;
  const [bin, meta] = await Promise.all([fetch(`${base}bank.bin`), fetch(`${base}bank.json`)]);
  if (!bin.ok || !meta.ok) throw new Error("Could not load the font bank");
  const { n } = (await meta.json()) as { n: number };
  return { buf: await bin.arrayBuffer(), n };
});
