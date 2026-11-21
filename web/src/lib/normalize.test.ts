import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { canvasToGlyph, fontFileToRefs, isEmpty } from "./normalize";
import { GUIDE_ROWS, SCALE } from "./glyphs";

const FIX = resolve(__dirname, "../../../tests/fixtures");
const loadBuf = (name: string) => {
  const b = readFileSync(resolve(FIX, name));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
};

function blankCanvas(): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = 256;
  c.height = 256;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, 256, 256);
  return c;
}

describe("canvasToGlyph", () => {
  it("keeps ink where the guides put it (no recentring)", () => {
    const c = blankCanvas();
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "#fff";
    ctx.fillRect(120, GUIDE_ROWS.cap * SCALE, 16, (GUIDE_ROWS.baseline - GUIDE_ROWS.cap) * SCALE);
    const { glyph, clipped } = canvasToGlyph(c);
    expect(glyph.length).toBe(4096);
    expect(clipped).toBe(false);
    const inkRows = new Set<number>();
    for (let i = 0; i < 4096; i++) if (glyph[i] > 0) inkRows.add(Math.floor(i / 64));
    expect(Math.min(...inkRows)).toBe(GUIDE_ROWS.cap);
    expect(Math.max(...inkRows)).toBe(GUIDE_ROWS.baseline - 1);
    expect(Math.min(...glyph)).toBe(-1);
    expect(Math.max(...glyph)).toBe(1);
  });

  it("flags a stroke that runs off the edge, without throwing", () => {
    const c = blankCanvas();
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "#fff";
    ctx.fillRect(200, 100, 56, 10);
    expect(canvasToGlyph(c).clipped).toBe(true);
  });

  it("treats an untouched canvas as empty", () => {
    expect(isEmpty(canvasToGlyph(blankCanvas()).glyph)).toBe(true);
  });
});

describe("fontFileToRefs", () => {
  it("renders H O n a g like the Python training renderer", () => {
    const refs = fontFileToRefs(loadBuf("Lato-Regular.ttf"));
    const expected = JSON.parse(readFileSync(resolve(FIX, "lato_refs.json"), "utf8")) as Record<string, number[]>;
    for (const c of ["H", "O", "n", "a", "g"] as const) {
      const exp = expected[c].map((v) => v / 127.5 - 1);
      let diff = 0;
      for (let i = 0; i < 4096; i++) diff += Math.abs(refs[c][i] - exp[i]);
      expect(diff / 4096, `letter ${c}`).toBeLessThan(0.1);
    }
  });

  it("rejects a file that is not a font", () => {
    const junk = new Uint8Array(2048).map((_, i) => (i * 37) % 256).buffer;
    expect(() => fontFileToRefs(junk)).toThrow("Could not read font file");
  });

  it("names the reference letters a font is missing", () => {
    expect(() => fontFileToRefs(loadBuf("Lato-no-g.ttf"))).toThrow("Font is missing: g");
  });
});
