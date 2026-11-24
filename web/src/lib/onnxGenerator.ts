// In-browser flow-matching sampler. The style encoder runs once; then the exported one-step U-Net is
// called `steps` times in an Euler loop: x <- x + dt * v(x, t, letter, style), from noise (t=0) to glyph (t=1).

import { fetchWithProgress } from "./download";
import type { Generator } from "./generator";
import { ALL_CHARS } from "./generator";
import { REF_CHARS, SIZE, charId, type GlyphBitmap, type RefSet } from "./glyphs";

export type Backend = "webgpu" | "wasm";
type TensorLike = { data: Float32Array | BigInt64Array | Uint8Array | ArrayLike<number>; dims: readonly number[] };
export interface SessionLike {
  run(feeds: Record<string, TensorLike>): Promise<Record<string, TensorLike>>;
}
export type SessionFactory = (url: string, ep: Backend) => Promise<SessionLike>;
type Assets = { nullStyle: Float32Array; steps: number; wasmSteps: number; cfg: number };

export type OnnxGenerator = Generator & {
  backend: () => Backend | null;
  /** Start downloading and loading the model now; Generate reuses the result. */
  warmUp: () => Promise<void>;
};

const PIX = SIZE * SIZE;
const clamp1 = (v: number) => Math.max(-1, Math.min(1, v));

/** The current image of every glyph being generated, after one Euler step. `glyphs[k]` is a copy for `chars[k]`. */
export type StepFrame = { step: number; steps: number; chars: number[]; glyphs: GlyphBitmap[] };

/** Seeded standard-normal noise (mulberry32 + Box–Muller). */
export function seededNormal(n: number, seed: number): Float32Array {
  let s = seed >>> 0;
  const rand = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = Math.max(rand(), 1e-12), v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

export function createOnnxGenerator(opts: {
  createSession: SessionFactory;
  loadAssets: () => Promise<Assets>;
  hasWebGPU: () => boolean;
  baseUrl?: string;
  onProgress?: (done: number, total: number) => void;
  /** Called after every step with the current images; the sampler waits for it, so the page can paint each frame. */
  onFrame?: (frame: StepFrame) => void | Promise<void>;
}): OnnxGenerator {
  const base = opts.baseUrl ?? "";
  let backend: Backend | null = null;
  let ready: Promise<{ enc: SessionLike; unet: SessionLike; assets: Assets }> | null = null;
  let calls = 0;

  const open = async () => {
    const assets = await opts.loadAssets();
    const eps: Backend[] = opts.hasWebGPU() ? ["webgpu", "wasm"] : ["wasm"];
    let lastErr: unknown = null;
    for (const ep of eps) {
      try {
        const [enc, unet] = await Promise.all([
          opts.createSession(`${base}models/encoder.onnx`, ep),
          opts.createSession(`${base}models/unet.onnx`, ep),
        ]);
        backend = ep;
        return { enc, unet, assets };
      } catch (e) {
        lastErr = e;
      }
    }
    throw new Error(`Could not start the model: ${String(lastErr)}`);
  };

  const load = () =>
    (ready ??= open().catch((e) => {
      ready = null; // allow a retry
      throw e;
    }));

  const gen = (async (refs: RefSet, chars: number[] = ALL_CHARS) => {
    const { enc, unet, assets } = await load();
    const seed = 1000 + calls++;

    const refData = new Float32Array(5 * PIX);
    REF_CHARS.forEach((c, k) => refData.set(refs[c], k * PIX));
    const { style } = await enc.run({
      refs: { data: refData, dims: [1, 5, 1, SIZE, SIZE] },
      ref_ids: { data: BigInt64Array.from(REF_CHARS.map((c) => BigInt(charId(c)))), dims: [1, 5] },
      ref_mask: { data: new Uint8Array(5).fill(1), dims: [1, 5] },
    });
    const n = chars.length;
    const styleB = new Float32Array(n * 256);
    const nullB = new Float32Array(n * 256);
    for (let i = 0; i < n; i++) {
      styleB.set(style.data as Float32Array, i * 256);
      nullB.set(assets.nullStyle, i * 256);
    }
    const ids = BigInt64Array.from(chars.map(BigInt));
    const x = seededNormal(n * PIX, seed);
    // Flow-matching paths are nearly straight, so the slow CPU fallback can take fewer, bigger steps.
    const steps = backend === "wasm" ? assets.wasmSteps : assets.steps;
    const dt = 1 / steps;
    for (let s = 0; s < steps; s++) {
      const t = new Float32Array(n).fill(s * dt);
      const feeds = (st: Float32Array) => ({
        x_t: { data: x, dims: [n, 1, SIZE, SIZE] },
        t: { data: t, dims: [n] },
        char_id: { data: ids, dims: [n] },
        style: { data: st, dims: [n, 256] },
      });
      const v = (await unet.run(feeds(styleB))).v.data as Float32Array;
      if (assets.cfg !== 1) {
        const vn = (await unet.run(feeds(nullB))).v.data as Float32Array;
        for (let i = 0; i < x.length; i++) x[i] += dt * (vn[i] + assets.cfg * (v[i] - vn[i]));
      } else {
        for (let i = 0; i < x.length; i++) x[i] += dt * v[i];
      }
      opts.onProgress?.(s + 1, steps);
      if (opts.onFrame) {
        const glyphs = chars.map((_, k) => x.slice(k * PIX, (k + 1) * PIX).map(clamp1));
        await opts.onFrame({ step: s + 1, steps, chars: [...chars], glyphs });
      }
    }
    const out: GlyphBitmap[] = [];
    for (let i = 0; i < n; i++) out.push(x.slice(i * PIX, (i + 1) * PIX).map(clamp1));
    return out;
  }) as OnnxGenerator;
  gen.backend = () => backend;
  gen.warmUp = async () => void (await load());
  return gen;
}

/** The real thing: ONNX Runtime Web, model files served from the site's /models folder. */
export function createBrowserOnnxGenerator(
  onProgress?: (done: number, total: number) => void,
  onDownload?: (loaded: number, total: number | null) => void,
  onFrame?: (frame: StepFrame) => void | Promise<void>,
): OnnxGenerator {
  const base = import.meta.env.BASE_URL;
  // Model files are downloaded once (a WebGPU -> WASM fallback reuses the bytes), and progress is
  // summed across files so the page can show one "x / y MB".
  const files = new Map<string, Promise<Uint8Array>>();
  const progress = new Map<string, { loaded: number; total: number | null }>();
  const report = () => {
    let loaded = 0, total = 0, known = true;
    for (const p of progress.values()) {
      loaded += p.loaded;
      if (p.total === null) known = false;
      else total += p.total;
    }
    onDownload?.(loaded, known ? total : null);
  };
  const bytes = (url: string) => {
    if (!files.has(url)) {
      progress.set(url, { loaded: 0, total: null });
      files.set(url, fetchWithProgress(url, (loaded, total) => (progress.set(url, { loaded, total }), report())).catch((e) => {
        files.delete(url);
        throw e;
      }));
    }
    return files.get(url)!;
  };
  return createOnnxGenerator({
    baseUrl: base,
    onProgress,
    onFrame,
    hasWebGPU: () => typeof navigator !== "undefined" && "gpu" in navigator,
    loadAssets: async () => {
      const [meta, nul] = await Promise.all([fetch(`${base}models/meta.json`), fetch(`${base}models/null_style.bin`)]);
      if (!meta.ok || !nul.ok) throw new Error("Could not download the model");
      const m = (await meta.json()) as { steps: number; wasmSteps?: number; cfg: number };
      return { nullStyle: new Float32Array(await nul.arrayBuffer()), steps: m.steps, wasmSteps: m.wasmSteps ?? m.steps, cfg: m.cfg };
    },
    createSession: async (url, ep) => {
      const ort = await import("onnxruntime-web/webgpu");
      const session = await ort.InferenceSession.create(await bytes(url), { executionProviders: [ep] });
      return {
        async run(feeds) {
          const t: Record<string, InstanceType<typeof ort.Tensor>> = {};
          for (const [k, v] of Object.entries(feeds)) {
            const type = v.data instanceof BigInt64Array ? "int64" : v.data instanceof Uint8Array ? "bool" : "float32";
            t[k] = new ort.Tensor(type, v.data as never, v.dims as number[]);
          }
          const out = await session.run(t);
          return Object.fromEntries(Object.entries(out).map(([k, v]) => [k, { data: v.data as Float32Array, dims: v.dims }]));
        },
      };
    },
  });
}
