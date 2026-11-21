import { describe, expect, it } from "vitest";
import { createOnnxGenerator, seededNormal, type SessionLike } from "./onnxGenerator";
import { REF_CHARS, type RefSet } from "./glyphs";

const refs = Object.fromEntries(REF_CHARS.map((c) => [c, new Float32Array(4096).fill(-1)])) as RefSet;

// Stand-in sessions: the encoder returns a fixed style; the U-Net returns velocity 0.5 everywhere,
// so after the Euler loop every pixel is noise + 0.5 (clamped). This checks our loop and plumbing, not ONNX Runtime.
function fakeSession(url: string): SessionLike {
  return {
    async run(feeds): Promise<Record<string, { data: Float32Array; dims: number[] }>> {
      if (url.endsWith("encoder.onnx")) return { style: { data: new Float32Array(256).fill(0.1), dims: [1, 256] } };
      const n = (feeds.x_t.dims as number[])[0];
      return { v: { data: new Float32Array(n * 4096).fill(0.5), dims: [n, 1, 64, 64] } };
    },
  };
}

const assets = async () => ({ nullStyle: new Float32Array(256), steps: 12, wasmSteps: 4, cfg: 1 });

describe("onnxGenerator", () => {
  it("falls back to WASM when WebGPU fails, and still returns 52 glyphs", async () => {
    const tried: string[] = [];
    const gen = createOnnxGenerator({
      hasWebGPU: () => true,
      loadAssets: assets,
      createSession: async (url, ep) => {
        tried.push(ep);
        if (ep === "webgpu") throw new Error("no adapter");
        return fakeSession(url);
      },
    });
    const out = await gen(refs);
    expect(gen.backend()).toBe("wasm");
    expect(tried).toContain("webgpu");
    expect(out).toHaveLength(52);
    for (const g of out) {
      expect(g).toHaveLength(4096);
      expect(Math.max(...g)).toBeLessThanOrEqual(1);
      expect(Math.min(...g)).toBeGreaterThanOrEqual(-1);
    }
  });

  it("skips WebGPU entirely when the browser has none", async () => {
    const tried: string[] = [];
    const gen = createOnnxGenerator({
      hasWebGPU: () => false,
      loadAssets: assets,
      createSession: async (url, ep) => (tried.push(ep), fakeSession(url)),
    });
    expect(await gen(refs, [0, 7])).toHaveLength(2);
    expect(tried.every((t) => t === "wasm")).toBe(true);
    expect(gen.backend()).toBe("wasm");
  });

  it("uses the shorter step count on WASM and the full one on WebGPU", async () => {
    for (const [ep, expected] of [["wasm", 4], ["webgpu", 12]] as const) {
      let unetCalls = 0;
      const gen = createOnnxGenerator({
        hasWebGPU: () => ep === "webgpu",
        loadAssets: assets,
        createSession: async (url) => {
          const s = fakeSession(url);
          return { run: async (f) => (url.endsWith("unet.onnx") && unetCalls++, s.run(f)) };
        },
      });
      await gen(refs, [0]);
      expect(unetCalls, ep).toBe(expected);
    }
  });

  it("applies guidance: a second U-Net call with the null style, combined as vn + cfg*(v - vn)", async () => {
    const nullStyle = new Float32Array(256).fill(-7);
    const styles: number[] = [];
    const gen = createOnnxGenerator({
      hasWebGPU: () => false,
      loadAssets: async () => ({ nullStyle, steps: 1, wasmSteps: 1, cfg: 2 }),
      createSession: async (url) => ({
        async run(feeds): Promise<Record<string, { data: Float32Array; dims: number[] }>> {
          if (url.endsWith("encoder.onnx")) return { style: { data: new Float32Array(256).fill(0.1), dims: [1, 256] } };
          const st = (feeds.style.data as Float32Array)[0];
          styles.push(st);
          // conditional velocity 0.5, unconditional 0.25 -> guided 0.25 + 2*(0.5-0.25) = 0.75
          const v = st === -7 ? 0.25 : 0.5;
          return { v: { data: new Float32Array(4096).fill(v), dims: [1, 1, 64, 64] } };
        },
      }),
    });
    const [g] = await gen(refs, [0]);
    expect(styles).toHaveLength(2);
    expect(styles).toContain(-7);
    const noise = (await import("./onnxGenerator")).seededNormal(4096, 1000);
    for (const i of [0, 100, 4095]) expect(g[i]).toBeCloseTo(Math.max(-1, Math.min(1, noise[i] + 0.75)), 5);
  });

  it("draws reproducible Gaussian noise from a seed", () => {
    const a = seededNormal(4096, 42), b = seededNormal(4096, 42), c = seededNormal(4096, 43);
    expect(Array.from(a)).toEqual(Array.from(b));
    expect(Array.from(a)).not.toEqual(Array.from(c));
    const mean = a.reduce((s, v) => s + v, 0) / a.length;
    const sd = Math.sqrt(a.reduce((s, v) => s + (v - mean) ** 2, 0) / a.length);
    expect(Math.abs(mean)).toBeLessThan(0.1);
    expect(Math.abs(sd - 1)).toBeLessThan(0.1);
  });
});
