import { useEffect, useState } from "react";

/** A text box rendered in the freshly built font. */
export default function TypeTest({ fontBuf, family }: { fontBuf: ArrayBuffer; family: string }) {
  const [loadedFamily, setLoadedFamily] = useState<string | null>(null);
  const [text, setText] = useState("Sphinx of black quartz, judge my vow");

  useEffect(() => {
    if (typeof FontFace === "undefined") return;
    const name = `${family}-${Math.random().toString(36).slice(2, 8)}`; // unique so every rebuild shows
    const face = new FontFace(name, fontBuf);
    let cancelled = false;
    face.load().then((f) => {
      if (cancelled) return;
      document.fonts.add(f);
      setLoadedFamily(name);
    });
    return () => {
      cancelled = true;
      document.fonts.delete(face);
    };
  }, [fontBuf, family]);

  return (
    <div className="typetest">
      <textarea
        aria-label="Type to test your font"
        value={text}
        onChange={(e) => setText(e.target.value)}
        style={{ fontFamily: loadedFamily ? `"${loadedFamily}", sans-serif` : "sans-serif" }}
        spellCheck={false}
      />
      <p className="hint">Only A–Z, a–z and space are in your font. Anything else falls back to a system font.</p>
    </div>
  );
}
