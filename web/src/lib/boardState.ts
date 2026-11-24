// Pure rules for what each of the 52 alphabet-board slots shows. The board component only draws these.

import { CHARS, type GlyphBitmap, type RefChar } from "./glyphs";
import type { StepFrame } from "./onnxGenerator";

export type Phase = "idle" | "denoising" | "tracing" | "done";

export type CellView =
  | { kind: "empty" }
  | { kind: "ref"; bitmap: GlyphBitmap }
  | { kind: "pending" } // generating, no frame yet: show static
  | { kind: "step"; bitmap: GlyphBitmap; step: number; steps: number }
  | { kind: "final"; bitmap: GlyphBitmap; outline: boolean };

export type BoardState = {
  refs: Partial<Record<RefChar, GlyphBitmap>>;
  glyphs: GlyphBitmap[] | null;
  /** CHARS indices being generated now; null when nothing is running. */
  running: number[] | null;
  frame: StepFrame | null;
  phase: Phase;
};

export function cellView(i: number, s: BoardState): CellView {
  const char = CHARS[i] as RefChar;
  const ref = s.refs[char];

  if (s.phase === "denoising" && s.running?.includes(i)) {
    // In a full run the user's own letters stay put: they are the model's input.
    if (s.running.length === CHARS.length && ref) return { kind: "ref", bitmap: ref };
    const k = s.frame ? s.frame.chars.indexOf(i) : -1;
    if (s.frame && k >= 0) return { kind: "step", bitmap: s.frame.glyphs[k], step: s.frame.step, steps: s.frame.steps };
    return { kind: "pending" };
  }
  if (s.glyphs) return { kind: "final", bitmap: s.glyphs[i], outline: s.phase === "tracing" };
  if (ref) return { kind: "ref", bitmap: ref };
  return { kind: "empty" };
}
