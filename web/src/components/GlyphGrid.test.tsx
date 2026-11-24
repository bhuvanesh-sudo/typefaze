import { cleanup, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { BoardState } from "../lib/boardState";
import GlyphGrid from "./GlyphGrid";

// jsdom has no canvas, so record what the board does to each canvas instead.
const ops = new WeakMap<HTMLCanvasElement, string[]>();
const log = (c: HTMLCanvasElement) => (ops.get(c) ?? ops.set(c, []).get(c)!);

const state = (h: Float32Array | undefined): BoardState => ({ refs: h ? { H: h } : {}, glyphs: null, running: null, frame: null, phase: "idle" });
const ink = (v: number) => new Float32Array(4096).fill(v);
const under = () => document.querySelector('[data-char="H"] canvas.under') as HTMLCanvasElement;

describe("GlyphGrid crossfade underlay", () => {
  beforeEach(() => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(function (this: HTMLCanvasElement) {
      return {
        clearRect: () => log(this).push("clear"),
        drawImage: () => log(this).push("draw"),
        createImageData: (w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4) }),
        putImageData: () => log(this).push("put"),
      } as unknown as CanvasRenderingContext2D;
    });
  });
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    // @ts-expect-error test cleanup of an optional browser API
    delete HTMLCanvasElement.prototype.animate;
  });

  it("leaves nothing behind when a slot is emptied (a cleared pad)", () => {
    const { rerender } = render(<GlyphGrid state={state(ink(1))} busy={false} onRegenerate={() => {}} />);
    rerender(<GlyphGrid state={state(undefined)} busy={false} onRegenerate={() => {}} />);
    expect(log(under()).at(-1)).toBe("clear");
  });

  it("does not keep the old picture when there is no fade (reduced motion or no animation support)", () => {
    const { rerender } = render(<GlyphGrid state={state(ink(1))} busy={false} onRegenerate={() => {}} />);
    rerender(<GlyphGrid state={state(ink(0.5))} busy={false} onRegenerate={() => {}} />);
    expect(log(under()).at(-1)).toBe("clear");
  });

  it("keeps the old picture only while the fade runs, then clears it", () => {
    const anims: { onfinish: null | (() => void); cancel: () => void }[] = [];
    // @ts-expect-error jsdom lacks the Web Animations API; provide a stand-in
    HTMLCanvasElement.prototype.animate = vi.fn(() => (anims.push({ onfinish: null, cancel: () => {} }), anims.at(-1)));
    const { rerender } = render(<GlyphGrid state={state(ink(1))} busy={false} onRegenerate={() => {}} />);
    rerender(<GlyphGrid state={state(ink(0.5))} busy={false} onRegenerate={() => {}} />);
    expect(log(under()).at(-1)).toBe("draw"); // mid-fade: old picture still underneath
    anims.at(-1)!.onfinish?.();
    expect(log(under()).at(-1)).toBe("clear"); // fade over: underlay gone
  });
});
