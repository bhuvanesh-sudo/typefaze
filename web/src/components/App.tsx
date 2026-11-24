import { useCallback, useEffect, useRef, useState } from "react";
import { buildFont } from "../lib/fontBuild";
import type { Phase } from "../lib/boardState";
import { ALL_CHARS, baselineGenerator, type Generator } from "../lib/generator";
import { createBrowserOnnxGenerator, type OnnxGenerator, type StepFrame } from "../lib/onnxGenerator";
import { nextPaint } from "../lib/paint";
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
  const glyphsRef = useRef<GlyphBitmap[] | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [running, setRunning] = useState<number[] | null>(null);
  const [frame, setFrame] = useState<StepFrame | null>(null);
  const [fontBuf, setFontBuf] = useState<ArrayBuffer | null>(null);
  const [busy, setBusy] = useState(false);
  const [name, setName] = useState("Mine");
  const [mode, setMode] = useState<"model" | "baseline">("model");
  const [progress, setProgress] = useState<string | null>(null);
  const onnx = useRef<OnnxGenerator | null>(null);
  const [modelStatus, setModelStatus] = useState<string>("Downloading the model…");
  const getOnnx = () =>
    (onnx.current ??= createBrowserOnnxGenerator(
      undefined,
      (loaded, total) => {
        const mb = (n: number) => (n / 1e6).toFixed(1);
        if (total && loaded >= total) setModelStatus("Model downloaded. Starting it up…");
        else setModelStatus(total ? `Downloading the model: ${mb(loaded)} / ${mb(total)} MB` : `Downloading the model: ${mb(loaded)} MB`);
      },
      // Show each real step on the board, and wait for it to be painted before the next step starts.
      (f) => {
        setFrame(f);
        return nextPaint();
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
    setFrame(null);
    // Compare mode has no model steps, so no slot pretends to be "starting from noise".
    setRunning(mode === "baseline" ? null : (chars ?? ALL_CHARS));
    setPhase("denoising");
    const started = performance.now();
    try {
      const out = await getGenerator()(refs as RefSet, chars);
      const secs = ((performance.now() - started) / 1000).toFixed(1);
      const where = mode === "baseline" ? "nearest font" : onnx.current?.backend() === "webgpu" ? "your GPU (WebGPU)" : "your CPU (WebGPU unavailable, so this is slower)";
      let merged = out;
      if (chars && glyphsRef.current) {
        merged = [...glyphsRef.current];
        chars.forEach((c, k) => (merged[c] = out[k]));
      }
      glyphsRef.current = merged;
      setGlyphs(merged);
      setRunning(null);
      setFrame(null);
      if (!chars) {
        // The tracing handoff is real: the font is built (outlines traced) right after the paint below.
        setPhase("tracing");
        await nextPaint();
      }
      setFontBuf(buildFont(merged, "TypeFaze Preview"));
      setPhase("done");
      setProgress(`Done in ${secs}s on ${where}.`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed");
      setProgress(null);
      setRunning(null);
      setFrame(null);
      setPhase(glyphsRef.current ? "done" : "idle");
    } finally {
      setBusy(false);
    }
  };

  // The preview font is built once per finished run; the chosen name is applied when downloading,
  // so typing a name doesn't re-trace 52 letters on every keystroke.
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

  const stepLine = (f: StepFrame | null) =>
    !f ? "Starting from noise…"
    : `Step ${f.step} of ${f.steps}. ${f.step >= f.steps ? "Sharpened. Tracing outlines next." : f.step / f.steps < 0.5 ? "Starting from noise…" : "Finding the shapes of your hand…"}`;
  const status =
    phase === "denoising" ? (mode === "baseline" ? "Finding the nearest font…" : stepLine(frame))
    : phase === "tracing" ? `Tracing ${running?.length === 1 ? "1 outline" : "52 outlines"}…`
    : !ready ? `Still to draw: ${missing.join(", ")}`
    : progress ?? "Ready when you are.";
  const active = phase === "denoising" || phase === "tracing";
  const step = fontBuf ? 3 : busy || ready ? 2 : 1;

  return (
    <main className="app">
      <header className="top">
        <div className="logo">TypeFaze</div>
        <nav className="trail" aria-label="Progress">
          {["Draw", "Generate", "Try it"].map((label, k) => (
            <span key={label} aria-current={step === k + 1 ? "step" : undefined}>
              {k + 1} {label}
            </span>
          ))}
        </nav>
      </header>

      <h1>
        Draw five letters.<span>Get a whole font.</span>
      </h1>
      <p className="lede">Trace H, O, n, a and g between the guides. A small model on your device draws the other 47.</p>

      <section aria-labelledby="refs-heading">
        <h2 id="refs-heading" className="sr-only">Draw these letters</h2>
        <p className="hint">
          Keep capitals between the top and bottom lines, lowercase between the middle and bottom lines, and let the tail
          of the g drop below.
        </p>
        <div className="pads">
          {REF_CHARS.map((c) => (
            <DrawPad key={c} letter={c} preset={presets[c] ?? null} onChange={onChange} />
          ))}
        </div>
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

      <section className="panel board" aria-labelledby="gen-heading">
        <div className="board-head">
          <div>
            <h2 id="gen-heading">Your alphabet</h2>
            <p className={`status${active ? " is-active" : ""}`} aria-live="polite">{status}</p>
            {mode === "model" && <p className="hint" aria-live="polite">{modelStatus}</p>}
          </div>
          <button type="button" className="primary" disabled={!ready || busy} onClick={() => void run()}>
            {busy ? "Generating…" : "Generate my font"}
          </button>
        </div>
        <GlyphGrid state={{ refs, glyphs, running, frame, phase }} busy={busy} onRegenerate={(i) => void run([i])} />
        <label className="toggle">
          <input type="checkbox" checked={mode === "baseline"} onChange={(e) => setMode(e.target.checked ? "baseline" : "model")} disabled={busy} />{" "}
          Compare: use the nearest existing font instead of the model
        </label>
        <label className="upload">
          …or start from letters in a font you already have:{" "}
          <input
            type="file"
            accept=".ttf,.otf,.woff"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void onUpload(f);
            }}
          />
        </label>
      </section>

      <section
        className={`panel test${fontBuf ? " is-ready" : ""}`}
        aria-labelledby="test-heading"
        aria-hidden={fontBuf ? undefined : true}
        {...(fontBuf ? {} : ({ inert: "" } as object))}
      >
        <h2 id="test-heading">Try it, then download</h2>
        {fontBuf ? (
          <>
            <TypeTest fontBuf={fontBuf} family="TypeFaze Preview" />
            <div className="row">
              <label>
                Font name: <input value={name} onChange={(e) => setName(e.target.value)} maxLength={24} />
              </label>
              <button type="button" className="primary" onClick={download}>
                Download .otf
              </button>
            </div>
          </>
        ) : (
          <p className="hint">Your font will show up here, ready to type with.</p>
        )}
      </section>
    </main>
  );
}
