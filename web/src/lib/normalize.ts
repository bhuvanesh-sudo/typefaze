// Turn a drawing canvas or an uploaded font into reference glyphs in the training convention.

import { parse, type Font } from "opentype.js";
import { BASELINE_ROW, CENTER_COL, EM_PX, REF_CHARS, SIZE, type GlyphBitmap, type RefSet } from "./glyphs";

/** Area-average a square, white-ink-on-black RGBA image down to 64x64 in [-1, 1]. */
export function imageDataToGlyph(data: Uint8ClampedArray, side: number): { glyph: GlyphBitmap; clipped: boolean } {
  const f = side / SIZE;
  const glyph = new Float32Array(SIZE * SIZE);
  let clipped = false;
  const lum = (x: number, y: number) => {
    const i = (y * side + x) * 4;
    return (Math.max(data[i], data[i + 1], data[i + 2]) / 255) * (data[i + 3] / 255);
  };
  for (let gy = 0; gy < SIZE; gy++) {
    for (let gx = 0; gx < SIZE; gx++) {
      let s = 0;
      for (let dy = 0; dy < f; dy++) for (let dx = 0; dx < f; dx++) s += lum(gx * f + dx, gy * f + dy);
      glyph[gy * SIZE + gx] = (s / (f * f)) * 2 - 1;
    }
  }
  for (let k = 0; k < side && !clipped; k++) {
    if (lum(k, 0) > 0.5 || lum(k, side - 1) > 0.5 || lum(0, k) > 0.5 || lum(side - 1, k) > 0.5) clipped = true;
  }
  return { glyph, clipped };
}

/** Read a 256x256 drawing canvas (white strokes on black). Position is kept as drawn: the guides define it. */
export function canvasToGlyph(canvas: HTMLCanvasElement): { glyph: GlyphBitmap; clipped: boolean } {
  const ctx = canvas.getContext("2d")!;
  return imageDataToGlyph(ctx.getImageData(0, 0, canvas.width, canvas.height).data, canvas.width);
}

/** Fewer than 20 inked pixels counts as "nothing drawn". */
export function isEmpty(g: GlyphBitmap): boolean {
  let n = 0;
  for (let i = 0; i < g.length; i++) if (g[i] > 0 && ++n >= 20) return false;
  return true;
}

/** Render one character of a parsed font with the training convention (em 48 px, baseline row 50, centred on advance). */
export function renderChar(font: Font, c: string): GlyphBitmap {
  const canvas = document.createElement("canvas");
  canvas.width = SIZE;
  canvas.height = SIZE;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, SIZE, SIZE);
  const glyph = font.charToGlyph(c);
  const advancePx = ((glyph.advanceWidth ?? 0) * EM_PX) / font.unitsPerEm;
  const path = glyph.getPath(CENTER_COL - advancePx / 2, BASELINE_ROW, EM_PX);
  path.fill = "#fff";
  path.draw(ctx);
  return imageDataToGlyph(ctx.getImageData(0, 0, SIZE, SIZE).data, SIZE).glyph;
}

export function parseFont(buf: ArrayBuffer): Font {
  try {
    return parse(buf);
  } catch {
    throw new Error("Could not read font file");
  }
}

/** Use an existing font's H O n a g as the reference letters. */
export function fontFileToRefs(buf: ArrayBuffer): RefSet {
  const font = parseFont(buf);
  const missing = REF_CHARS.filter((c) => font.charToGlyphIndex(c) === 0);
  if (missing.length) throw new Error(`Font is missing: ${missing.join(", ")}`);
  return Object.fromEntries(REF_CHARS.map((c) => [c, renderChar(font, c)])) as RefSet;
}
