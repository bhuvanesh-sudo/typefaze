// @vitest-environment node
// Real-model check, run on demand: TYPEFAZE_REAL=1 npx vitest run src/realmodel.test.ts
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import * as ort from "onnxruntime-web";
import { expect, it } from "vitest";
import { createOnnxGenerator } from "./lib/onnxGenerator";
import { REF_CHARS, type RefSet } from "./lib/glyphs";

it.runIf(process.env.TYPEFAZE_REAL)("trained model runs through the TS sampler (WASM)", async () => {
  const M = resolve(__dirname, "../public/models");
  const exp = JSON.parse(readFileSync(resolve(__dirname, "../../tests/fixtures/lato_refs.json"), "utf8")) as Record<string, number[]>;
  const refs = Object.fromEntries(REF_CHARS.map((c) => [c, Float32Array.from(exp[c], (v) => v / 127.5 - 1)])) as RefSet;
  const gen = createOnnxGenerator({
    hasWebGPU: () => false,
    loadAssets: async () => ({ nullStyle: (() => { const b = readFileSync(`${M}/null_style.bin`); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); })(), steps: 12, wasmSteps: 4, cfg: 1 }),
    createSession: async (url) => {
      const s = await ort.InferenceSession.create(resolve(M, url.split("/").pop()!), { executionProviders: ["wasm"] });
      return {
        async run(feeds) {
          const t: Record<string, ort.Tensor> = {};
          for (const [k, v] of Object.entries(feeds)) {
            const type = v.data instanceof BigInt64Array ? "int64" : v.data instanceof Uint8Array ? "bool" : "float32";
            t[k] = new ort.Tensor(type, v.data as never, v.dims as number[]);
          }
          const o = await s.run(t);
          return Object.fromEntries(Object.entries(o).map(([k, v]) => [k, { data: v.data as Float32Array, dims: v.dims }]));
        },
      };
    },
  });
  const chars = [0, 1, 6, 17, 30, 32, 41, 50]; // A B G R e g p y
  const out = await gen(refs, chars);
  expect(gen.backend()).toBe("wasm");
  writeFileSync("/tmp/ts_glyphs.json", JSON.stringify(out.map((g) => Array.from(g))));
}, 600_000);
