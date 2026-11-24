import { describe, expect, it } from "vitest";
import { outlinePathData } from "./outline";

describe("outlinePathData", () => {
  it("has no outline for a blank glyph", () => {
    expect(outlinePathData(new Float32Array(4096).fill(-1))).toBe("");
  });

  it("traces a filled rectangle back to its own pixel box", () => {
    const g = new Float32Array(4096).fill(-1);
    for (let y = 20; y < 50; y++) for (let x = 26; x < 38; x++) g[y * 64 + x] = 1;
    const d = outlinePathData(g);
    expect(d.startsWith("M")).toBe(true);
    expect(d.endsWith("Z")).toBe(true);
    const nums = (d.match(/-?\d+(\.\d+)?/g) ?? []).map(Number);
    expect(nums.length).toBeGreaterThanOrEqual(8);
    for (let i = 0; i < nums.length; i += 2) {
      expect(nums[i]).toBeGreaterThanOrEqual(25);
      expect(nums[i]).toBeLessThanOrEqual(39);
      expect(nums[i + 1]).toBeGreaterThanOrEqual(19);
      expect(nums[i + 1]).toBeLessThanOrEqual(51);
    }
  });
});
