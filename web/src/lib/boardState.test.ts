import { describe, expect, it } from "vitest";
import { cellView, type BoardState } from "./boardState";
import { CHARS } from "./glyphs";
import type { StepFrame } from "./onnxGenerator";

const blank = () => new Float32Array(4096).fill(-1);
const ALL = Array.from(CHARS, (_, i) => i);
const H = CHARS.indexOf("H"); // 7
const A = 0;
const B = 1;

const base = (over: Partial<BoardState> = {}): BoardState => ({ refs: {}, glyphs: null, running: null, frame: null, phase: "idle", ...over });
const frame = (chars: number[], step = 2, steps = 4): StepFrame => ({ step, steps, chars, glyphs: chars.map(blank) });
const done = () => ALL.map(blank);

describe("cellView", () => {
  it("shows empty slots before anything is drawn", () => {
    for (const i of ALL) expect(cellView(i, base()).kind).toBe("empty");
  });

  it("drawn and preset reference letters fill their own slot, and clearing empties it again", () => {
    expect(cellView(H, base({ refs: { H: blank() } })).kind).toBe("ref");
    expect(cellView(A, base({ refs: { H: blank() } })).kind).toBe("empty");
    expect(cellView(H, base({ refs: { H: undefined } })).kind).toBe("empty");
  });

  it("full run: references stay, the other 47 are pending, then show the frame", () => {
    const s = base({ refs: { H: blank() }, running: ALL, phase: "denoising" });
    expect(cellView(H, s).kind).toBe("ref");
    expect(cellView(A, s).kind).toBe("pending");
    const withFrame = { ...s, frame: frame(ALL, 2, 4) };
    expect(cellView(A, withFrame)).toMatchObject({ kind: "step", step: 2, steps: 4 });
    expect(cellView(H, withFrame).kind).toBe("ref");
  });

  it("works for any step count", () => {
    const s = base({ running: ALL, phase: "denoising", frame: frame(ALL, 1, 12) });
    expect(cellView(A, s)).toMatchObject({ kind: "step", step: 1, steps: 12 });
  });

  it("tracing shows outlines on every final slot, done does not", () => {
    expect(cellView(A, base({ glyphs: done(), phase: "tracing" }))).toMatchObject({ kind: "final", outline: true });
    expect(cellView(A, base({ glyphs: done(), phase: "done" }))).toMatchObject({ kind: "final", outline: false });
  });

  it("a single-letter regenerate only touches that slot", () => {
    const s = base({ glyphs: done(), running: [5], phase: "denoising" });
    expect(cellView(5, s).kind).toBe("pending");
    expect(cellView(4, s).kind).toBe("final");
    const withFrame = { ...s, frame: frame([5]) };
    expect(cellView(5, withFrame).kind).toBe("step");
    expect(cellView(4, withFrame).kind).toBe("final");
  });

  it("a single regenerate of a reference letter still shows frames, not the drawing", () => {
    const s = base({ refs: { H: blank() }, glyphs: done(), running: [H], phase: "denoising", frame: frame([H]) });
    expect(cellView(H, s).kind).toBe("step");
    expect(cellView(B, s).kind).toBe("final");
  });
});
