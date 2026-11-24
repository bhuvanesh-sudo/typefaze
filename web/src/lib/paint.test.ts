import { afterEach, describe, expect, it, vi } from "vitest";
import { nextPaint } from "./paint";

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("nextPaint", () => {
  it("still resolves when animation frames never fire (a hidden tab)", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("requestAnimationFrame", () => 0);
    let done = false;
    void nextPaint().then(() => (done = true));
    await vi.advanceTimersByTimeAsync(500);
    expect(done).toBe(true);
  });

  it("resolves after a frame in the normal case", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("requestAnimationFrame", (cb: () => void) => setTimeout(cb, 16));
    let done = false;
    void nextPaint().then(() => (done = true));
    await vi.advanceTimersByTimeAsync(5);
    expect(done).toBe(false);
    await vi.advanceTimersByTimeAsync(30);
    expect(done).toBe(true);
  });
});
