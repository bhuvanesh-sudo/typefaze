// 52 glyph bitmaps -> an installable OpenType (CFF) font.

import { Font, Glyph, Path } from "opentype.js";
import { CHARS, type GlyphBitmap } from "./glyphs";
import { traceGlyph } from "./trace";

const SIDE_BEARING = 60;

/** Font-internal names must be plain ASCII (PostScript names even stricter), or some systems refuse the font. */
export function safeFamilyName(name: string): string {
  const ascii = name.replace(/[^\x20-\x7e]/g, "").replace(/[()[\]{}<>/%\\]/g, "").replace(/\s+/g, " ").trim();
  return ascii && ascii !== "TypeFaze" ? ascii : "TypeFaze";
}
const BLANK_ADVANCE = 250;

export function buildFont(glyphs: GlyphBitmap[], familyName: string): ArrayBuffer {
  if (glyphs.length !== CHARS.length) throw new Error(`expected ${CHARS.length} glyphs, got ${glyphs.length}`);
  const notdef = new Glyph({ name: ".notdef", unicode: 0, advanceWidth: 500, path: new Path() });
  const space = new Glyph({ name: "space", unicode: 32, advanceWidth: BLANK_ADVANCE, path: new Path() });

  const letters = glyphs.map((g, i) => {
    const path = traceGlyph(g);
    let advanceWidth = BLANK_ADVANCE;
    if (path.commands.length) {
      const bb = path.getBoundingBox();
      const shift = SIDE_BEARING - bb.x1;
      for (const c of path.commands) {
        if ("x" in c) c.x += shift;
        if ("x1" in c) c.x1 += shift;
        if ("x2" in c) c.x2 += shift;
      }
      advanceWidth = Math.round(bb.x2 - bb.x1 + 2 * SIDE_BEARING);
    }
    return new Glyph({ name: CHARS[i], unicode: CHARS.charCodeAt(i), advanceWidth, path });
  });

  const family = safeFamilyName(familyName);
  const font = new Font({
    familyName: family,
    fullName: `${family} Regular`,
    postScriptName: `${family.replace(/[^A-Za-z0-9-]/g, "") || "TypeFaze"}-Regular`,
    styleName: "Regular",
    unitsPerEm: 1000,
    ascender: 1000,
    descender: -300,
    designer: "TypeFaze",
    glyphs: [notdef, space, ...letters],
  });
  return font.toArrayBuffer();
}
