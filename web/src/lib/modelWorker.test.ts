// The page talks to the model through messages, so loading and generating run on a background
// thread. These tests wire the page-side client straight to the worker-side handler (no real thread).
import { afterEach, describe, expect, it, vi } from "vitest";
import { createWorkerHandler, type FromWorker, type ToWorker } from "./modelWorkerCore";
import { createWorkerOnnxGenerator, type WorkerLike } from "./modelWorkerClient";
import type { OnnxGenerator } from "./onnxGenerator";
import { REF_CHARS, type RefSet } from "./glyphs";

const refs = Object.fromEntries(REF_CHARS.map((c) => [c, new Float32Array(4096).fill(-1)])) as RefSet;

type Hooks = { onStep?: (d: number, t: number) => void; onDownload?: (l: number, t: number | null) => void };

function stubGenerator(hooks: Hooks, opts: { failWarmUp?: boolean; failGenerate?: boolean } = {}): OnnxGenerator {
  const g = (async (_r: RefSet, chars?: number[]) => {
    if (opts.failGenerate) throw new Error("boom");
    hooks.onStep?.(1, 2);
    hooks.onStep?.(2, 2);
    return (chars ?? [0, 1, 2]).map((c) => new Float32Array(4096).fill(c / 100));
  }) as OnnxGenerator;
  g.backend = () => "webgpu";
  g.warmUp = async () => {
    hooks.onDownload?.(10, 20);
    if (opts.failWarmUp) throw new Error("offline");
  };
  return g;
}

function pair(opts?: Parameters<typeof stubGenerator>[1]) {
  const listeners: ((e: { data: FromWorker }) => void)[] = [];
  const handle = createWorkerHandler(
    (hooks) => stubGenerator(hooks, opts),
    (msg) => queueMicrotask(() => listeners.forEach((l) => l({ data: msg }))),
  );
  const worker = {
    postMessage: (msg: ToWorker) => void queueMicrotask(() => void handle(msg)),
    addEventListener: (t: string, fn: (e: { data: FromWorker }) => void) => void (t === "message" && listeners.push(fn)),
  } as WorkerLike;
  return worker;
}

describe("model worker protocol", () => {
  it("generates in the worker and forwards progress and backend to the page", async () => {
    const steps: number[] = [];
    const gen = createWorkerOnnxGenerator(pair(), (d) => steps.push(d));
    const out = await gen(refs, [5, 7]);
    expect(out).toHaveLength(2);
    expect(out[1][0]).toBeCloseTo(0.07);
    expect(steps).toEqual([1, 2]);
    expect(gen.backend()).toBe("webgpu");
  });

  it("warmUp resolves once the worker has loaded, and reports download progress", async () => {
    const dl: [number, number | null][] = [];
    const gen = createWorkerOnnxGenerator(pair(), undefined, (l, t) => dl.push([l, t]));
    await gen.warmUp();
    expect(dl).toEqual([[10, 20]]);
  });

  it("a failed warmUp rejects with the worker's message", async () => {
    await expect(createWorkerOnnxGenerator(pair({ failWarmUp: true })).warmUp()).rejects.toThrow("offline");
  });

  it("keeps simultaneous requests apart", async () => {
    const gen = createWorkerOnnxGenerator(pair());
    const [a, b] = await Promise.all([gen(refs, [1]), gen(refs, [2, 3])]);
    expect(a).toHaveLength(1);
    expect(b).toHaveLength(2);
    expect(a[0][0]).toBeCloseTo(0.01);
  });

  it("a generation error rejects that request only", async () => {
    await expect(createWorkerOnnxGenerator(pair({ failGenerate: true }))(refs)).rejects.toThrow("boom");
  });
});

describe("worker crashes", () => {
  it("rejects waiting requests and warm-up with the crash message instead of hanging", async () => {
    const handlers: Record<string, ((e: unknown) => void)[]> = {};
    const worker = {
      postMessage: () => {},
      addEventListener: (t: string, fn: (e: unknown) => void) => void (handlers[t] ??= []).push(fn),
    } as unknown as WorkerLike;
    const gen = createWorkerOnnxGenerator(worker);
    const warm = gen.warmUp();
    const job = gen(refs);
    handlers.error.forEach((h) => h({ message: "Failed to load ORT" }));
    await expect(warm).rejects.toThrow("The model worker crashed: Failed to load ORT");
    await expect(job).rejects.toThrow("The model worker crashed: Failed to load ORT");
  });
});

describe("createModelGenerator", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("starts a module worker when the browser has Worker", async () => {
    const made: { url: string; type?: string }[] = [];
    class FakeWorker {
      constructor(url: URL | string, opts?: { type?: string }) {
        made.push({ url: String(url), type: opts?.type });
      }
      postMessage() {}
      addEventListener() {}
    }
    vi.stubGlobal("Worker", FakeWorker);
    const { createModelGenerator } = await import("./modelWorkerClient");
    const gen = createModelGenerator({ inPage: () => stubGenerator({}) });
    expect(made).toHaveLength(1);
    expect(made[0].url).toContain("model.worker");
    expect(made[0].type).toBe("module");
    expect(typeof gen.warmUp).toBe("function");
  });

  it("falls back to running on the page when Worker is unavailable", async () => {
    vi.stubGlobal("Worker", undefined);
    const { createModelGenerator } = await import("./modelWorkerClient");
    let fallback = 0;
    const gen = createModelGenerator({ inPage: () => (fallback++, stubGenerator({})) });
    expect(fallback).toBe(1);
    expect(gen.backend()).toBe("webgpu");
  });
});
