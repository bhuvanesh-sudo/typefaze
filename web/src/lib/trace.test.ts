import { describe, expect, it } from "vitest";
import { traceGlyph } from "./trace";

const blank = () => new Float32Array(4096).fill(-1);

describe("traceGlyph", () => {
  it("maps a square at rows 30–50 to font units above the baseline", () => {
    const g = blank();
    for (let y = 30; y < 50; y++) for (let x = 22; x < 42; x++) g[y * 64 + x] = 1;
    const bb = traceGlyph(g).getBoundingBox();
    expect(bb.y1).toBeGreaterThan(-30);
    expect(bb.y1).toBeLessThan(30);
    expect(bb.y2).toBeGreaterThan(((50 - 30) * 1000) / 48 - 30);
    expect(bb.y2).toBeLessThan(((50 - 30) * 1000) / 48 + 30);
    expect(bb.x2 - bb.x1).toBeGreaterThan((20 * 1000) / 48 - 30);
  });

  it("returns an empty path for a blank glyph", () => {
    expect(traceGlyph(blank()).commands).toHaveLength(0);
  });

  it("does not throw on a solid glyph", () => {
    const path = traceGlyph(new Float32Array(4096).fill(1));
    expect(path.commands.length).toBeGreaterThan(0);
  });

  it("winds holes opposite to outer contours (an O stays hollow)", () => {
    const g = blank();
    for (let y = 10; y < 50; y++)
      for (let x = 12; x < 52; x++) {
        const inner = y >= 20 && y < 40 && x >= 22 && x < 42;
        g[y * 64 + x] = inner ? -1 : 1;
      }
    const path = traceGlyph(g);
    const contours: number[] = [];
    let pts: [number, number][] = [];
    const flush = () => {
      if (pts.length > 2) {
        let a = 0;
        for (let i = 0; i < pts.length; i++) {
          const [x1, y1] = pts[i];
          const [x2, y2] = pts[(i + 1) % pts.length];
          a += x1 * y2 - x2 * y1;
        }
        contours.push(a / 2);
      }
      pts = [];
    };
    for (const c of path.commands) {
      if (c.type === "M") { flush(); pts.push([c.x, c.y]); }
      else if (c.type === "L" || c.type === "Q") pts.push([c.x, c.y]);
    }
    flush();
    expect(contours).toHaveLength(2);
    expect(Math.sign(contours[0])).toBe(-Math.sign(contours[1]));
  });
});
