import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const buildSpy = vi.hoisted(() => ({ calls: [] as string[] }));

vi.mock("../lib/onnxGenerator", () => ({
  createBrowserOnnxGenerator: () => {
    const g = async (_refs: unknown, chars?: number[]) =>
      (chars ?? Array.from({ length: 52 }, (_, i) => i)).map(() => {
        const a = new Float32Array(4096).fill(-1);
        for (let y = 20; y < 50; y++) for (let x = 26; x < 38; x++) a[y * 64 + x] = 1;
        return a;
      });
    g.backend = () => "wasm";
    return g;
  },
}));

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

describe("App generate -> name -> download", () => {
  const created: string[] = [];
  const revoked: string[] = [];
  beforeEach(() => {
    buildSpy.calls.length = 0;
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
    for (const l of ["H", "O", "n", "a", "g"]) drawStroke(screen.getByLabelText(`Draw ${l}`) as HTMLCanvasElement);
    await act(async () => fireEvent.click(screen.getByRole("button", { name: /generate/i })));
    const buildsAfterGenerate = buildSpy.calls.length;
    expect(buildsAfterGenerate).toBe(1);

    const name = screen.getByLabelText(/font name/i);
    for (const v of ["M", "My", "MyF", "MyFo", "MyFon"]) fireEvent.change(name, { target: { value: v } });
    expect(buildSpy.calls.length).toBe(buildsAfterGenerate);

    await act(async () => fireEvent.click(screen.getByRole("button", { name: /download/i })));
    expect(buildSpy.calls.at(-1)).toBe("TypeFaze MyFon");
    expect(created).toHaveLength(1);
    await act(async () => new Promise((r) => setTimeout(r, 1100)));
    expect(revoked).toEqual(created);
  });

  it("says so when a drawn letter is too small to read", async () => {
    render(<App />);
    drawStroke(screen.getByLabelText("Draw H") as HTMLCanvasElement, 4);
    expect(screen.getByText(/too small/i)).toBeInTheDocument();
  });
});
