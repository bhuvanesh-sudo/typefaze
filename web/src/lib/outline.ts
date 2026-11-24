// The traced outline of a glyph as SVG path data, in the glyph's own 64x64 pixel space.
// Uses the same tracer as the font export, so what the board shows is what lands in the .otf.

import { BASELINE_ROW, CENTER_COL, EM_PX, type GlyphBitmap } from "./glyphs";
import { traceGlyph } from "./trace";

const UNITS_PER_PX = 1000 / EM_PX; // same scale as trace.ts

export function outlinePathData(g: GlyphBitmap): string {
  const r = (v: number) => Math.round(v * 10) / 10;
  const px = (x: number) => r(x / UNITS_PER_PX + CENTER_COL);
  const py = (y: number) => r(BASELINE_ROW - y / UNITS_PER_PX);
  const parts: string[] = [];
  for (const c of traceGlyph(g).commands) {
    switch (c.type) {
      case "M": parts.push(`M${px(c.x)} ${py(c.y)}`); break;
      case "L": parts.push(`L${px(c.x)} ${py(c.y)}`); break;
      case "Q": parts.push(`Q${px(c.x1)} ${py(c.y1)} ${px(c.x)} ${py(c.y)}`); break;
      case "C": parts.push(`C${px(c.x1)} ${py(c.y1)} ${px(c.x2)} ${py(c.y2)} ${px(c.x)} ${py(c.y)}`); break;
      case "Z": parts.push("Z"); break;
    }
  }
  return parts.join("");
}
