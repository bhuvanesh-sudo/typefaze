import { useEffect, useRef } from "react";
import { GUIDE_ROWS, SCALE, SIZE, type GlyphBitmap, type RefChar } from "../lib/glyphs";
import { canvasToGlyph, isEmpty } from "../lib/normalize";

const SIDE = SIZE * SCALE; // 256
const STROKE = 14; // px on the 256 canvas, about a regular-weight stem at 64 px

type Props = {
  letter: RefChar;
  /** Set to show a glyph from an uploaded font instead of a drawing. */
  preset?: GlyphBitmap | null;
  onChange: (letter: RefChar, glyph: GlyphBitmap | null, clipped: boolean, tooSmall?: boolean) => void;
};

function paintBlank(ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, SIDE, SIDE);
}

/** One drawing square for one reference letter, with typographic guide lines in a separate layer. */
export default function DrawPad({ letter, preset, onChange }: Props) {
  const ink = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const last = useRef<[number, number] | null>(null);

  useEffect(() => {
    const ctx = ink.current?.getContext("2d");
    if (!ctx) return;
    paintBlank(ctx);
    if (preset) {
      const img = ctx.createImageData(SIDE, SIDE);
      for (let y = 0; y < SIDE; y++)
        for (let x = 0; x < SIDE; x++) {
          const v = Math.round(((preset[Math.floor(y / SCALE) * SIZE + Math.floor(x / SCALE)] + 1) / 2) * 255);
          const i = (y * SIDE + x) * 4;
          img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
          img.data[i + 3] = 255;
        }
      ctx.putImageData(img, 0, 0);
    }
  }, [preset]);

  const pos = (e: React.PointerEvent<HTMLCanvasElement>): [number, number] => {
    const r = e.currentTarget.getBoundingClientRect();
    return [((e.clientX - r.left) * SIDE) / (r.width || SIDE), ((e.clientY - r.top) * SIDE) / (r.height || SIDE)];
  };

  const line = (a: [number, number], b: [number, number]) => {
    const ctx = ink.current!.getContext("2d")!;
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = STROKE;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(...a);
    ctx.lineTo(...b);
    ctx.stroke();
  };

  const finish = () => {
    if (!drawing.current) return;
    drawing.current = false;
    last.current = null;
    const { glyph, clipped } = canvasToGlyph(ink.current!);
    const empty = isEmpty(glyph);
    const anyInk = glyph.some((v) => v > -0.9);
    onChange(letter, empty ? null : glyph, clipped, empty && anyInk);
  };

  const clear = () => {
    paintBlank(ink.current!.getContext("2d")!);
    onChange(letter, null, false);
  };

  return (
    <figure className="pad">
      <div className="pad-stack">
        <canvas
          ref={ink}
          width={SIDE}
          height={SIDE}
          aria-label={`Draw ${letter}`}
          onPointerDown={(e) => {
            e.currentTarget.setPointerCapture?.(e.pointerId);
            drawing.current = true;
            const p = pos(e);
            last.current = p;
            line(p, p);
          }}
          onPointerMove={(e) => {
            if (!drawing.current) return;
            const p = pos(e);
            line(last.current ?? p, p);
            last.current = p;
          }}
          onPointerUp={finish}
          onPointerLeave={finish}
        />
        <svg className="guides" viewBox={`0 0 ${SIZE} ${SIZE}`} aria-hidden="true">
          {Object.entries(GUIDE_ROWS).map(([name, row]) => (
            <line key={name} className={`guide guide-${name}`} x1={0} x2={SIZE} y1={row} y2={row} />
          ))}
        </svg>
        <span className="pad-letter" aria-hidden="true">
          {letter}
        </span>
      </div>
      <figcaption>
        <button type="button" onClick={clear}>
          Clear {letter}
        </button>
      </figcaption>
    </figure>
  );
}
