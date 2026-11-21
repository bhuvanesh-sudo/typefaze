// Page side: looks exactly like the in-page OnnxGenerator, but the model runs in a Web Worker,
// so loading and generating never block drawing, clicking or scrolling.

import type { GlyphBitmap, RefSet } from "./glyphs";
import type { FromWorker, ToWorker } from "./modelWorkerCore";
import type { Backend, OnnxGenerator } from "./onnxGenerator";

export type WorkerLike = {
  postMessage: (msg: ToWorker) => void;
  addEventListener(type: "message", fn: (e: { data: FromWorker }) => void): void;
  addEventListener(type: "error", fn: (e: { message?: string }) => void): void;
};

export function createWorkerOnnxGenerator(
  worker: WorkerLike,
  onProgress?: (done: number, total: number) => void,
  onDownload?: (loaded: number, total: number | null) => void,
): OnnxGenerator {
  let backend: Backend | null = null;
  let nextId = 0;
  const pending = new Map<number, { resolve: (g: GlyphBitmap[]) => void; reject: (e: Error) => void }>();
  let warm: { promise: Promise<void>; resolve: () => void; reject: (e: Error) => void } | null = null;

  worker.addEventListener("message", ({ data }) => {
    switch (data.type) {
      case "download":
        onDownload?.(data.loaded, data.total);
        break;
      case "step":
        onProgress?.(data.done, data.total);
        break;
      case "ready":
        backend = data.backend;
        warm?.resolve();
        break;
      case "warmupError":
        warm?.reject(new Error(data.message));
        warm = null; // allow a retry
        break;
      case "result":
        backend = data.backend;
        pending.get(data.id)?.resolve(data.glyphs);
        pending.delete(data.id);
        break;
      case "error":
        pending.get(data.id)?.reject(new Error(data.message));
        pending.delete(data.id);
        break;
    }
  });

  // A crash inside the worker (e.g. a script that fails to load) never sends a message back,
  // so without this every request would wait forever.
  worker.addEventListener("error", (e) => {
    const err = new Error(`The model worker crashed: ${e.message || "unknown error"}`);
    for (const p of pending.values()) p.reject(err);
    pending.clear();
    warm?.reject(err);
    warm = null;
  });

  const gen = ((refs: RefSet, chars?: number[]) =>
    new Promise<GlyphBitmap[]>((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject });
      worker.postMessage({ type: "generate", id, refs, chars });
    })) as OnnxGenerator;
  gen.backend = () => backend;
  gen.warmUp = () => {
    if (!warm) {
      let resolve!: () => void, reject!: (e: Error) => void;
      const promise = new Promise<void>((res, rej) => ((resolve = res), (reject = rej)));
      warm = { promise, resolve, reject };
      worker.postMessage({ type: "warmup" });
    }
    return warm.promise;
  };
  return gen;
}

/** The model generator the app uses: a background worker when possible, the page itself otherwise. */
export function createModelGenerator(opts: {
  onProgress?: (done: number, total: number) => void;
  onDownload?: (loaded: number, total: number | null) => void;
  inPage: () => OnnxGenerator;
}): OnnxGenerator {
  if (typeof Worker === "undefined") return opts.inPage();
  // Written literally so Vite recognises and bundles the worker.
  const worker = new Worker(new URL("./model.worker.ts", import.meta.url), { type: "module" });
  return createWorkerOnnxGenerator(worker as unknown as WorkerLike, opts.onProgress, opts.onDownload);
}
