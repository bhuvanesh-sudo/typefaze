import { parse } from "opentype.js";
import { describe, expect, it } from "vitest";
import { buildFont } from "./fontBuild";

describe("buildFont", () => {
  it("builds a parseable font with 52 letters, .notdef and space, tolerating a blank glyph", () => {
    const glyphs = Array.from({ length: 52 }, (_, i) => {
      const g = new Float32Array(4096).fill(-1);
      if (i !== 5) for (let y = 20; y < 50; y++) for (let x = 24; x < 40; x++) g[y * 64 + x] = 1;
      return g;
    });
    const font = parse(buildFont(glyphs, "Test Face"));
    expect(font.numGlyphs).toBe(54);
    expect(font.unitsPerEm).toBe(1000);
    expect(font.ascender).toBe(1000);
    expect(font.descender).toBe(-300);
    expect(font.charToGlyph("A").advanceWidth).toBeGreaterThan(0);
    expect(font.charToGlyph("F").advanceWidth).toBe(250); // index 5 is blank
    expect(font.charToGlyph(" ").advanceWidth).toBe(250);
    // opentype.js 2.x nests names by platform
    expect((font.names as unknown as { unicode: { fontFamily: { en: string } } }).unicode.fontFamily.en).toBe("Test Face");
  });

  it("keeps internal font names to safe ASCII whatever the user types", () => {
    const glyphs = Array.from({ length: 52 }, () => new Float32Array(4096).fill(-1));
    const font = parse(buildFont(glyphs, "TypeFaze 😀 日本 (x)/y"));
    const names = (font.names as unknown as { unicode: Record<string, { en: string }> }).unicode;
    expect(names.postScriptName.en).toMatch(/^[A-Za-z0-9-]+$/);
    for (const key of ["fontFamily", "fullName", "postScriptName"]) expect(names[key].en).toMatch(/^[\x20-\x7e]+$/);
    expect(names.fontFamily.en).toContain("TypeFaze");
  });

  it("falls back to a default name when nothing usable is left", () => {
    const glyphs = Array.from({ length: 52 }, () => new Float32Array(4096).fill(-1));
    const names = (parse(buildFont(glyphs, "😀😀")).names as unknown as { unicode: Record<string, { en: string }> }).unicode;
    expect(names.fontFamily.en).toBe("TypeFaze");
  });
});
