import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { buildFont } from "../lib/fontBuild";
import { baselineGenerator, type Generator } from "../lib/generator";
import { createBrowserOnnxGenerator, type OnnxGenerator } from "../lib/onnxGenerator";
import { REF_CHARS, type GlyphBitmap, type RefChar, type RefSet } from "../lib/glyphs";
import { fontFileToRefs } from "../lib/normalize";
import DrawPad from "./DrawPad";
import GlyphGrid from "./GlyphGrid";
import TypeTest from "./TypeTest";

type Partial5 = Partial<Record<RefChar, GlyphBitmap>>;

export default function App() {
  const [refs, setRefs] = useState<Partial5>({});
  const [presets, setPresets] = useState<Partial5>({});
  const [clipped, setClipped] = useState<Partial<Record<RefChar, boolean>>>({});
  const [tooSmall, setTooSmall] = useState<Partial<Record<RefChar, boolean>>>({});
  const [error, setError] = useState<string | null>(null);
  const [glyphs, setGlyphs] = useState<GlyphBitmap[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [name, setName] = useState("Mine");
  const [mode, setMode] = useState<"model" | "baseline">("model");
  const [progress, setProgress] = useState<string | null>(null);
  const onnx = useRef<OnnxGenerator | null>(null);
  const [modelStatus, setModelStatus] = useState<string>("Downloading the model…");
  const getOnnx = () =>
    (onnx.current ??= createBrowserOnnxGenerator(
      (done, total) => setProgress(`Step ${done} of ${total}`),
      (loaded, total) => {
        const mb = (n: number) => (n / 1e6).toFixed(1);
        if (total && loaded >= total) setModelStatus("Model downloaded. Starting it up…");
        else setModelStatus(total ? `Downloading the model: ${mb(loaded)} / ${mb(total)} MB` : `Downloading the model: ${mb(loaded)} MB`);
      },
    ));
  const getGenerator = (): Generator => (mode === "baseline" ? baselineGenerator : getOnnx());

  // Start fetching the model the moment the page opens, so it's usually ready by the time
  // five letters are drawn. Generate waits for the same download instead of starting another.
  useEffect(() => {
    let live = true;
    getOnnx()
      .warmUp()
      .then(() => live && setModelStatus("Model ready."))
      .catch(() => live && setModelStatus("Couldn't download the model yet. It will retry when you press Generate."));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onChange = useCallback((letter: RefChar, glyph: GlyphBitmap | null, wasClipped: boolean, small = false) => {
    setTooSmall((t) => ({ ...t, [letter]: small }));
    setRefs((r) => {
      const next = { ...r };
      if (glyph) next[letter] = glyph;
      else delete next[letter];
      return next;
    });
    setClipped((c) => ({ ...c, [letter]: wasClipped }));
  }, []);

  const onUpload = async (file: File) => {
    setError(null);
    try {
      const set: RefSet = fontFileToRefs(await file.arrayBuffer());
      setRefs({ ...set });
      setPresets({ ...set });
      setClipped({});
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not read font file");
    }
  };

  const missing = REF_CHARS.filter((c) => !refs[c]);
  const ready = missing.length === 0;

  const run = async (chars?: number[]) => {
    if (!ready) return;
    setBusy(true);
    setError(null);
    setProgress(null);
    const started = performance.now();
    try {
      const out = await getGenerator()(refs as RefSet, chars);
      const secs = ((performance.now() - started) / 1000).toFixed(1);
      const where = mode === "baseline" ? "nearest font" : onnx.current?.backend() === "webgpu" ? "your GPU (WebGPU)" : "your CPU (WebGPU unavailable, so this is slower)";
      setProgress(`Done in ${secs}s on ${where}.`);
      setGlyphs((prev) => {
        if (!chars || !prev) return out;
        const next = [...prev];
        chars.forEach((c, k) => (next[c] = out[k]));
        return next;
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed");
      setProgress(null);
    } finally {
      setBusy(false);
    }
  };

  // The preview font depends only on the glyphs; the chosen name is applied when downloading,
  // so typing a name doesn't re-trace 52 letters on every keystroke.
  const fontBuf = useMemo(() => (glyphs ? buildFont(glyphs, "TypeFaze Preview") : null), [glyphs]);
  const download = () => {
    if (!glyphs) return;
    const label = name.trim() || "Mine";
    const url = URL.createObjectURL(new Blob([buildFont(glyphs, `TypeFaze ${label}`)], { type: "font/otf" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `TypeFaze-${label.replace(/[^A-Za-z0-9-]/g, "") || "Mine"}.otf`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const clippedLetters = REF_CHARS.filter((c) => clipped[c]);
  const smallLetters = REF_CHARS.filter((c) => tooSmall[c]);

  return (
    <main className="app">
      <header>
        <h1>TypeFaze</h1>
        <p>Draw five letters. Get a whole font.</p>
      </header>

      <section aria-labelledby="refs-heading">
        <h2 id="refs-heading">1. Draw these letters</h2>
        <p className="hint">
          Keep capitals between the top and bottom lines, lowercase between the middle and bottom lines, and let the tail
          of the g drop below.
        </p>
        <div className="pads">
          {REF_CHARS.map((c) => (
            <DrawPad key={c} letter={c} preset={presets[c] ?? null} onChange={onChange} />
          ))}
        </div>
        <label className="upload">
          …or use letters from a font you already have:{" "}
          <input
            type="file"
            accept=".ttf,.otf,.woff"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void onUpload(f);
            }}
          />
        </label>
        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}
        {smallLetters.length > 0 && (
          <p className="warning">
            Your {smallLetters.join(", ")} was too small to read. Draw it bigger, using the guide lines.
          </p>
        )}
        {clippedLetters.length > 0 && (
          <p className="warning">
            Part of your {clippedLetters.join(", ")} ran off the edge and was cut off. Try drawing it a little smaller.
          </p>
        )}
      </section>

      <section aria-labelledby="gen-heading">
        <h2 id="gen-heading">2. Generate</h2>
        <button type="button" className="primary" disabled={!ready || busy} onClick={() => void run()}>
          {busy ? "Generating…" : "Generate"}
        </button>
        <label className="toggle">
          <input type="checkbox" checked={mode === "baseline"} onChange={(e) => setMode(e.target.checked ? "baseline" : "model")} disabled={busy} />{" "}
          Compare: use the nearest existing font instead of the model
        </label>
        {!ready && <p className="hint">Still to draw: {missing.join(", ")}</p>}
        {mode === "model" && <p className="hint" aria-live="polite">{modelStatus}</p>}
        {progress && <p className="hint" aria-live="polite">{progress}</p>}
        {glyphs && (
          <>
            <p className="hint">Click any letter to try it again.</p>
            <GlyphGrid glyphs={glyphs} busy={busy} onRegenerate={(i) => void run([i])} />
          </>
        )}
      </section>

      {fontBuf && (
        <section aria-labelledby="test-heading">
          <h2 id="test-heading">3. Try it and download</h2>
          <TypeTest fontBuf={fontBuf} family="TypeFaze Preview" />
          <div className="row">
            <label>
              Font name: <input value={name} onChange={(e) => setName(e.target.value)} maxLength={24} />
            </label>
            <button type="button" className="primary" onClick={download}>
              Download .otf
            </button>
          </div>
        </section>
      )}
    </main>
  );
}
