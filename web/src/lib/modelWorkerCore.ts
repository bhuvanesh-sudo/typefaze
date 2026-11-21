// Worker side of the page <-> model protocol. Kept free of worker globals so it can be tested directly.

import type { RefSet } from "./glyphs";
import type { Backend, OnnxGenerator } from "./onnxGenerator";

export type ToWorker = { type: "warmup" } | { type: "generate"; id: number; refs: RefSet; chars?: number[] };

export type FromWorker =
  | { type: "download"; loaded: number; total: number | null }
  | { type: "step"; done: number; total: number }
  | { type: "ready"; backend: Backend | null }
  | { type: "warmupError"; message: string }
  | { type: "result"; id: number; glyphs: Float32Array[]; backend: Backend | null }
  | { type: "error"; id: number; message: string };

type Hooks = { onStep: (done: number, total: number) => void; onDownload: (loaded: number, total: number | null) => void };

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

export function createWorkerHandler(
  make: (hooks: Hooks) => OnnxGenerator,
  post: (msg: FromWorker, transfer?: Transferable[]) => void,
): (msg: ToWorker) => Promise<void> {
  const gen = make({
    onStep: (done, total) => post({ type: "step", done, total }),
    onDownload: (loaded, total) => post({ type: "download", loaded, total }),
  });
  return async (msg) => {
    if (msg.type === "warmup") {
      try {
        await gen.warmUp();
        post({ type: "ready", backend: gen.backend() });
      } catch (e) {
        post({ type: "warmupError", message: message(e) });
      }
      return;
    }
    try {
      const glyphs = await gen(msg.refs, msg.chars);
      post({ type: "result", id: msg.id, glyphs, backend: gen.backend() }, glyphs.map((g) => g.buffer as ArrayBuffer));
    } catch (e) {
      post({ type: "error", id: msg.id, message: message(e) });
    }
  };
}
