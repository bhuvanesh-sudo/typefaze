// End-to-end without a browser: font upload -> nearest-font baseline over the real bank -> installable font.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parse } from "opentype.js";
import { expect, it } from "vitest";
import { buildFont } from "./lib/fontBuild";
import { createBaselineGenerator } from "./lib/generator";
import { fontFileToRefs } from "./lib/normalize";

const read = (p: string) => {
  const b = readFileSync(p);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
};

it("upload -> baseline -> .otf works with the real 200-font bank", async () => {
  const refs = fontFileToRefs(read(resolve(__dirname, "../../tests/fixtures/Lato-Regular.ttf")));
  const meta = JSON.parse(readFileSync(resolve(__dirname, "../public/bank.json"), "utf8")) as { n: number };
  const gen = createBaselineGenerator(async () => ({ buf: read(resolve(__dirname, "../public/bank.bin")), n: meta.n }));
  const glyphs = await gen(refs);
  expect(glyphs).toHaveLength(52);
  const font = parse(buildFont(glyphs, "TypeFaze E2E"));
  expect(font.numGlyphs).toBe(54);
  expect(font.charToGlyph("g").path.commands.length).toBeGreaterThan(0);
});
