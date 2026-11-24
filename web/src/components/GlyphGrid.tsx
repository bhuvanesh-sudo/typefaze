import { useEffect, useMemo, useRef } from "react";
import { cellView, type BoardState, type CellView } from "../lib/boardState";
import { CHARS, SIZE, type GlyphBitmap } from "../lib/glyphs";
import { outlinePathData } from "../lib/outline";

const INK = [241, 235, 226]; // --ink

/** Ink as alpha over transparent, so the slot's own background (and static, and outlines) show through. */
function paint(canvas: HTMLCanvasElement | null, bitmap: GlyphBitmap | null) {
  const ctx = canvas?.getContext("2d");
  if (!canvas || !ctx) return;
  if (!bitmap) {
    ctx.clearRect(0, 0, SIZE, SIZE);
    return;
  }
  const img = ctx.createImageData(SIZE, SIZE);
  for (let i = 0; i < bitmap.length; i++) {
    img.data[i * 4] = INK[0];
    img.data[i * 4 + 1] = INK[1];
    img.data[i * 4 + 2] = INK[2];
    img.data[i * 4 + 3] = Math.round(((Math.max(-1, Math.min(1, bitmap[i])) + 1) / 2) * 255);
  }
  ctx.putImageData(img, 0, 0);
}

const reduceMotion = () => typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;

function Cell({ i, view, busy, onClick }: { i: number; view: CellView; busy: boolean; onClick: () => void }) {
  const char = CHARS[i];
  const under = useRef<HTMLCanvasElement>(null);
  const over = useRef<HTMLCanvasElement>(null);
  const bitmap = "bitmap" in view ? view.bitmap : null;
  const outline = view.kind === "final" && view.outline;
  const path = useMemo(() => (outline && bitmap ? outlinePathData(bitmap) : ""), [outline, bitmap]);

  // Keep the previous picture underneath only while the new one fades in, then clear it.
  // Ink is drawn as alpha, so anything left on the underlay would show through for good.
  const fade = useRef<Animation | null>(null);
  useEffect(() => {
    const top = over.current;
    const ctxUnder = under.current?.getContext("2d");
    const clearUnder = () => ctxUnder?.clearRect(0, 0, SIZE, SIZE);
    fade.current?.cancel();
    fade.current = null;
    if (top && ctxUnder) {
      clearUnder();
      ctxUnder.drawImage(top, 0, 0);
    }
    paint(top, bitmap);
    if (bitmap && top && typeof top.animate === "function" && !reduceMotion()) {
      const a = top.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 260, easing: "cubic-bezier(.22, 1, .36, 1)" });
      fade.current = a;
      a.onfinish = () => {
        if (fade.current === a) {
          clearUnder();
          fade.current = null;
        }
      };
    } else {
      clearUnder();
    }
  }, [bitmap]);

  const state = view.kind === "final" && view.outline ? "outline" : view.kind;
  const inner = (
    <>
      <canvas ref={under} width={SIZE} height={SIZE} className="under" aria-hidden="true" />
      <canvas ref={over} width={SIZE} height={SIZE} className="over" aria-hidden="true" />
      {path && (
        <svg className="trace" viewBox={`0 0 ${SIZE} ${SIZE}`} aria-hidden="true">
          <path d={path} />
        </svg>
      )}
      <span className="label" aria-hidden="true">{char}</span>
    </>
  );
  if (view.kind === "final") {
    return (
      <button type="button" className="cell" data-char={char} data-state={state} onClick={onClick} disabled={busy} aria-label={`Try ${char} again`} title={`Try ${char} again`}>
        {inner}
      </button>
    );
  }
  return (
    <div className="cell" data-char={char} data-state={state} role="img" aria-label={char}>
      {inner}
    </div>
  );
}

/** The alphabet board: all 52 slots, from the first stroke to the finished font. Clicking a finished letter asks for a fresh attempt at just that one. */
export default function GlyphGrid({ state, busy, onRegenerate }: { state: BoardState; busy: boolean; onRegenerate: (i: number) => void }) {
  return (
    <div className="grid">
      {Array.from(CHARS, (_, i) => (
        <Cell key={CHARS[i]} i={i} view={cellView(i, state)} busy={busy} onClick={() => onRegenerate(i)} />
      ))}
    </div>
  );
}
