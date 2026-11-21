import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { createBaselineGenerator, unpackGlyph } from "./generator";
import { REF_CHARS, charId, type RefSet } from "./glyphs";

const FIX = resolve(__dirname, "../../../tests/fixtures");
const bankBuf = () => {
  const b = readFileSync(resolve(FIX, "lato_bank.bin"));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
};

describe("baselineGenerator", () => {
  it("returns 52 glyphs of 4096 values from a one-font bank", async () => {
    const gen = createBaselineGenerator(async () => ({ buf: bankBuf(), n: 1 }));
    const bank = bankBuf();
    const refs = Object.fromEntries(REF_CHARS.map((c) => [c, unpackGlyph(bank, 0, charId(c))])) as RefSet;
    const out = await gen(refs);
    expect(out).toHaveLength(52);
    for (const g of out) expect(g).toHaveLength(4096);
    expect(Array.from(out[charId("A")])).toEqual(Array.from(unpackGlyph(bank, 0, charId("A"))));
  });

  it("loads the bank only once", async () => {
    let loads = 0;
    const gen = createBaselineGenerator(async () => {
      loads++;
      return { buf: bankBuf(), n: 1 };
    });
    const refs = Object.fromEntries(REF_CHARS.map((c) => [c, new Float32Array(4096).fill(-1)])) as RefSet;
    await gen(refs);
    await gen(refs);
    expect(loads).toBe(1);
  });
});
