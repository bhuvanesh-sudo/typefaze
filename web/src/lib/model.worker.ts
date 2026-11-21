// Runs the model off the page's main thread. All the logic lives in modelWorkerCore (tested).
import { createWorkerHandler, type FromWorker, type ToWorker } from "./modelWorkerCore";
import { createBrowserOnnxGenerator } from "./onnxGenerator";

const scope = self as unknown as {
  postMessage: (msg: FromWorker, transfer?: Transferable[]) => void;
  addEventListener: (type: "message", fn: (e: MessageEvent<ToWorker>) => void) => void;
};

const handle = createWorkerHandler(
  (hooks) => createBrowserOnnxGenerator(hooks.onStep, hooks.onDownload),
  (msg, transfer) => scope.postMessage(msg, transfer ?? []),
);
scope.addEventListener("message", (e) => void handle(e.data));
