/**
 * Resolves once the browser has had a chance to paint what React just committed.
 * Hidden tabs never fire animation frames, so a short timeout also resolves it: a run must not stall in the background.
 */
export function nextPaint(): Promise<void> {
  return new Promise((resolve) => {
    let settled = false;
    const done = () => {
      if (settled) return;
      settled = true;
      clearTimeout(fallback);
      resolve();
    };
    const fallback = setTimeout(done, 100);
    if (typeof requestAnimationFrame === "undefined" || (typeof document !== "undefined" && document.hidden)) setTimeout(done, 0);
    else requestAnimationFrame(() => setTimeout(done, 0));
  });
}
