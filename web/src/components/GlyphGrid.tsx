import { useEffect, useRef } from "react";
import { CHARS, SIZE, type GlyphBitmap } from "../lib/glyphs";

function GlyphCell({ glyph, char, busy, onClick }: { glyph: GlyphBitmap; char: string; busy: boolean; onClick: () => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(SIZE, SIZE);
    for (let i = 0; i < glyph.length; i++) {
      const v = Math.round(((glyph[i] + 1) / 2) * 255);
      img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [glyph]);
  return (
    <button type="button" className="cell" onClick={onClick} disabled={busy} aria-label={`Regenerate ${char}`} title={`Click to regenerate ${char}`}>
      <canvas ref={ref} width={SIZE} height={SIZE} />
      <span className="label">{char}</span>
    </button>
  );
}

/** All 52 generated letters. Clicking one asks for a fresh attempt at just that letter. */
export default function GlyphGrid({ glyphs, busy, onRegenerate }: { glyphs: GlyphBitmap[]; busy: boolean; onRegenerate: (i: number) => void }) {
  return (
    <div className="grid">
      {glyphs.map((g, i) => (
        <GlyphCell key={CHARS[i]} glyph={g} char={CHARS[i]} busy={busy} onClick={() => onRegenerate(i)} />
      ))}
    </div>
  );
}
