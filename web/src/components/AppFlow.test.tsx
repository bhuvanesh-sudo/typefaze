import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const buildSpy = vi.hoisted(() => ({ calls: [] as string[] }));

const warm = vi.hoisted(() => ({ calls: 0, release: null as null | (() => void) }));

// Controls for the fake model: emit a frame per run, hold the run open, or fail it.
const model = vi.hoisted(() => ({
  frames: false,
  step: 1,
  steps: 4,
  gate: null as null | Promise<void>,
  release: null as null | (() => void),
  fail: false,
}));

const stripe = () => {
  const a = new Float32Array(4096).fill(-1);
  for (let y = 20; y < 50; y++) for (let x = 26; x < 38; x++) a[y * 64 + x] = 1;
  return a;
};

vi.mock("../lib/onnxGenerator", () => ({
  createBrowserOnnxGenerator: (
    _onStep: unknown,
    onDownload?: (loaded: number, total: number | null) => void,
    onFrame?: (f: unknown) => void | Promise<void>,
  ) => {
    const g = async (_refs: unknown, chars?: number[]) => {
      const list = chars ?? Array.from({ length: 52 }, (_, i) => i);
      if (model.fail) throw new Error("model blew up");
      if (model.frames) await onFrame?.({ step: model.step, steps: model.steps, chars: list, glyphs: list.map(stripe) });
      if (model.gate) await model.gate;
      return list.map(stripe);
    };
    g.backend = () => "wasm";
    g.warmUp = () => {
      warm.calls++;
      onDownload?.(5_000_000, 27_300_000);
      return new Promise<void>((r) => (warm.release = () => (onDownload?.(27_300_000, 27_300_000), r())));
    };
    return g;
  },
}));

vi.mock("../lib/generator", async (orig) => {
  const real = await orig<typeof import("../lib/generator")>();
  return {
    ...real,
    baselineGenerator: async (_r: unknown, chars?: number[]) => {
      if (model.gate) await model.gate;
      return (chars ?? real.ALL_CHARS).map(stripe);
    },
  };
});

vi.mock("../lib/fontBuild", async (orig) => {
  const real = await orig<typeof import("../lib/fontBuild")>();
  return { ...real, buildFont: (g: Float32Array[], name: string) => (buildSpy.calls.push(name), real.buildFont(g, name)) };
});

import App from "./App";

function drawStroke(canvas: HTMLCanvasElement, len = 120) {
  canvas.getBoundingClientRect = () =>
    ({ left: 0, top: 0, width: 256, height: 256, right: 256, bottom: 256, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect;
  fireEvent.pointerDown(canvas, { clientX: 128, clientY: 70, pointerId: 1 });
  fireEvent.pointerMove(canvas, { clientX: 128, clientY: 70 + len, pointerId: 1 });
  fireEvent.pointerUp(canvas, { clientX: 128, clientY: 70 + len, pointerId: 1 });
}

const drawAll = () => {
  for (const l of ["H", "O", "n", "a", "g"]) drawStroke(screen.getByLabelText(`Draw ${l}`) as HTMLCanvasElement);
};
const clickGenerate = () => act(async () => void fireEvent.click(screen.getByRole("button", { name: /generate/i })));
const cell = (c: string) => document.querySelector(`[data-char="${c}"]`) as HTMLElement;
const hold = () => (model.gate = new Promise<void>((r) => (model.release = r)));

describe("App generate -> name -> download", () => {
  const created: string[] = [];
  const revoked: string[] = [];
  beforeEach(() => {
    warm.calls = 0;
    warm.release = null;
    buildSpy.calls.length = 0;
    Object.assign(model, { frames: false, step: 1, steps: 4, gate: null, release: null, fail: false });
    URL.createObjectURL = vi.fn(() => {
      const u = `blob:${created.length}`;
      created.push(u);
      return u;
    });
    URL.revokeObjectURL = vi.fn((u: string) => void revoked.push(u));
    HTMLAnchorElement.prototype.click = vi.fn();
  });
  afterEach(() => cleanup());

  it("does not rebuild the font on every keystroke, and frees each download link", async () => {
    render(<App />);
    drawAll();
    await clickGenerate();
    await waitFor(() => expect(buildSpy.calls.length).toBe(1));
    const buildsAfterGenerate = buildSpy.calls.length;

    const name = screen.getByLabelText(/font name/i);
    for (const v of ["M", "My", "MyF", "MyFo", "MyFon"]) fireEvent.change(name, { target: { value: v } });
    expect(buildSpy.calls.length).toBe(buildsAfterGenerate);

    await act(async () => fireEvent.click(screen.getByRole("button", { name: /download/i })));
    expect(buildSpy.calls.at(-1)).toBe("TypeFaze MyFon");
    expect(created).toHaveLength(1);
    await act(async () => new Promise((r) => setTimeout(r, 1100)));
    expect(revoked).toEqual(created);
  });

  it("starts downloading the model as soon as the page opens, with progress", async () => {
    render(<App />);
    expect(warm.calls).toBe(1);
    expect(screen.getByText(/5\.0 \/ 27\.3 MB/)).toBeInTheDocument();
    await act(async () => warm.release?.());
    expect(screen.getByText(/model ready/i)).toBeInTheDocument();
  });

  it("a pad shows as filled once it has a letter, and not after Clear", () => {
    render(<App />);
    const pad = () => screen.getByLabelText("Draw H").closest("figure")!;
    expect(pad()).not.toHaveAttribute("data-filled", "true");
    drawStroke(screen.getByLabelText("Draw H") as HTMLCanvasElement);
    expect(pad()).toHaveAttribute("data-filled", "true");
    fireEvent.click(screen.getByRole("button", { name: "Clear H" }));
    expect(pad()).not.toHaveAttribute("data-filled", "true");
  });

  it("says so when a drawn letter is too small to read", async () => {
    render(<App />);
    drawStroke(screen.getByLabelText("Draw H") as HTMLCanvasElement, 4);
    expect(screen.getByText(/too small/i)).toBeInTheDocument();
  });
});

describe("the alphabet board", () => {
  beforeEach(() => {
    warm.calls = 0;
    buildSpy.calls.length = 0;
    Object.assign(model, { frames: false, step: 1, steps: 4, gate: null, release: null, fail: false });
    URL.createObjectURL = vi.fn(() => "blob:x");
    URL.revokeObjectURL = vi.fn();
  });
  afterEach(() => cleanup());

  const states = () => Array.from(document.querySelectorAll("[data-char]")).map((e) => e.getAttribute("data-state"));
  const allFinal = () => states().length === 52 && states().every((s) => s === "final");

  it("shows 52 slots from the start, and a drawn reference letter lands in its own slot", () => {
    render(<App />);
    expect(states()).toHaveLength(52);
    expect(states().every((s) => s === "empty")).toBe(true);
    drawStroke(screen.getByLabelText("Draw H") as HTMLCanvasElement);
    expect(cell("H")).toHaveAttribute("data-state", "ref");
    expect(cell("A")).toHaveAttribute("data-state", "empty");
  });

  it("shows the model's frames while generating, then the final letters", async () => {
    model.frames = true;
    hold();
    render(<App />);
    drawAll();
    await clickGenerate();
    await waitFor(() => expect(cell("A")).toHaveAttribute("data-state", "step"));
    expect(cell("H")).toHaveAttribute("data-state", "ref");
    await act(async () => model.release!());
    await waitFor(() => expect(allFinal()).toBe(true));
  });

  it("copes with a different step count", async () => {
    model.frames = true;
    model.steps = 12;
    hold();
    render(<App />);
    drawAll();
    await clickGenerate();
    await waitFor(() => expect(cell("A")).toHaveAttribute("data-state", "step"));
    await act(async () => model.release!());
    await waitFor(() => expect(allFinal()).toBe(true));
  });

  it("compare mode still fills the board, with no frames", async () => {
    render(<App />);
    drawAll();
    fireEvent.click(screen.getByLabelText(/nearest existing font/i));
    await clickGenerate();
    await waitFor(() => expect(allFinal()).toBe(true));
    expect(states()).not.toContain("pending");
  });

  it("a failed generation restores the board and re-enables Generate", async () => {
    model.fail = true;
    render(<App />);
    drawAll();
    await clickGenerate();
    expect(await screen.findByRole("alert")).toHaveTextContent("model blew up");
    expect(states()).not.toContain("pending");
    expect(states()).not.toContain("step");
    expect(cell("H")).toHaveAttribute("data-state", "ref");
    expect(screen.getByRole("button", { name: /generate/i })).toBeEnabled();
  });

  it("regenerating one letter animates only that slot", async () => {
    render(<App />);
    drawAll();
    await clickGenerate();
    await waitFor(() => expect(allFinal()).toBe(true));

    model.frames = true;
    hold();
    await act(async () => void fireEvent.click(screen.getByRole("button", { name: "Try B again" })));
    await waitFor(() => expect(cell("B")).toHaveAttribute("data-state", "step"));
    expect(cell("A")).toHaveAttribute("data-state", "final");
    await act(async () => model.release!());
    await waitFor(() => expect(cell("B")).toHaveAttribute("data-state", "final"));
  });

  it("compare mode shows no static and says what it is really doing", async () => {
    hold();
    render(<App />);
    drawAll();
    fireEvent.click(screen.getByLabelText(/nearest existing font/i));
    await clickGenerate();
    expect(await screen.findByText("Finding the nearest font…")).toBeInTheDocument();
    expect(states()).not.toContain("pending");
    expect(screen.queryByText(/starting from noise/i)).toBeNull();
    await act(async () => model.release!());
    await waitFor(() => expect(allFinal()).toBe(true));
  });

  it("says which letter is missing again after a pad is cleared following a run", async () => {
    render(<App />);
    drawAll();
    await clickGenerate();
    await waitFor(() => expect(allFinal()).toBe(true));
    fireEvent.click(screen.getByRole("button", { name: "Clear H" }));
    expect(screen.getByText("Still to draw: H")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /generate/i })).toBeDisabled();
  });

  it("only the main button is named like 'generate'", async () => {
    render(<App />);
    expect(screen.getAllByRole("button", { name: /generate/i })).toHaveLength(1);
    drawAll();
    await clickGenerate();
    await waitFor(() => expect(allFinal()).toBe(true));
    expect(screen.getAllByRole("button", { name: /generate/i })).toHaveLength(1);
  });
});

describe("status line and type test", () => {
  beforeEach(() => {
    warm.calls = 0;
    buildSpy.calls.length = 0;
    Object.assign(model, { frames: false, step: 1, steps: 4, gate: null, release: null, fail: false });
    URL.createObjectURL = vi.fn(() => "blob:x");
    URL.revokeObjectURL = vi.fn();
  });
  afterEach(() => cleanup());

  it("reads the step from the model's real frame", async () => {
    Object.assign(model, { frames: true, step: 2, steps: 4 });
    hold();
    render(<App />);
    drawAll();
    await clickGenerate();
    expect(await screen.findByText("Step 2 of 4. Finding the shapes of your hand…")).toBeInTheDocument();
    await act(async () => model.release!());
  });

  it("uses the real step total, not a fixed 4", async () => {
    Object.assign(model, { frames: true, step: 3, steps: 12 });
    hold();
    render(<App />);
    drawAll();
    await clickGenerate();
    expect(await screen.findByText(/Step 3 of 12/)).toBeInTheDocument();
    await act(async () => model.release!());
  });

  it("keeps the type test out of reach until a font exists", async () => {
    render(<App />);
    expect(screen.queryByRole("textbox", { name: /font name/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /download/i })).toBeNull();
    drawAll();
    await clickGenerate();
    expect(await screen.findByRole("textbox", { name: /font name/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /download/i })).toBeEnabled();
  });
});
