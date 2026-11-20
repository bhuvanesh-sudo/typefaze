# TypeFaze — Design

Date: 2026-11-20 · Build window: Sat 21 – Sun 22 Nov 2026 (two full days)

## Intent

A visitor draws a handful of letters in the browser; a generative model trained from scratch by the author produces the rest of the Latin alphabet in that style; the visitor downloads an installable font file. Everything runs on the visitor's device: no inference server.

Why: a portfolio project for ML roles that is visually impressive *and* technically defensible in an interview. Every component (data pipeline, model, training, export, evaluation) must be the author's own work and explainable. It already appears on the author's resume as "In Progress" with these claims, which the build must make true:

- few-shot: a handful of drawn letters → rest of the alphabet, exported as an installable OpenType font
- compact diffusion-family model trained **from scratch** on OFL-licensed Google Fonts glyphs
- exported to ONNX for fully in-browser inference (WebGPU)
- evaluated on legibility (OCR-style accuracy) and style consistency against a font-retrieval baseline

## Goals (by Sunday night)

1. End-to-end flow works in a browser: draw → generate 52 glyphs → trace to outlines → download `.otf` → install and type with it.
2. Generator is a small model trained from scratch, running client-side via ONNX Runtime Web.
3. An evaluation table on held-out fonts: model vs. retrieval baseline, with honest failure cases.
4. Public GitHub repo with a README (results, failure gallery, limitations) and the demo deployed as a static site (GitHub Pages).

## Non-goals (this weekend)

- Mobile/touch polish (it may work; not tested or tuned)
- Anything beyond A–Z and a–z (no digits, punctuation, non-Latin scripts)
- Direct vector (Bézier) generation; kerning; hinting; high-resolution glyphs
- Few-step distillation; fine-tuning pretrained models

## Anti-goals (failure even if it "works")

- Any server-side inference dependency
- Results in the README that were not actually measured, or a demo that only looks good with cherry-picked inputs presented as typical
- Training on fonts that are not OFL-licensed

## Constraints

- **Time:** two days, one person. Training gets roughly one overnight run plus slack for one retry.
- **Compute:** Kaggle free GPU (T4/P100, ~12 h sessions, ~30 h/week). Sessions can die: checkpoint frequently.
- **Browser:** total model download ideally < 30 MB; must work without WebGPU (WASM fallback), since the model is small enough for CPU.
- **Author experience:** assume this is the author's first end-to-end generative training run. Prefer simple, well-trodden components over clever ones.
- **Author's existing stack:** Python, React + Vite, TypeScript.

## Approaches considered

1. **Small conditional flow-matching model trained from scratch, in-browser (chosen).** ~10–30M params at 64×64. Matches the resume claims; fully explainable. Risk: training must work with little slack.
2. **Fine-tune an existing font model (e.g. FontDiffuser).** Better quality sooner, but too large for the browser (needs a server) and not "from scratch". Rejected.
3. **Retrieval only (nearest existing Google Font).** Trivial and reliable but not generative. Rejected as the product; **kept as the evaluation baseline and as the Saturday-morning placeholder generator** so the full flow works before the model is trained.

Chosen: 1, built in the order below so a working demo exists even if training fails.

## Design

### Data
- Source: the `google/fonts` GitHub repo, `ofl/` directory only (OFL licence). Use static font files; for variable fonts, instantiate the default instance only.
- Keep fonts that contain all 52 glyphs A–Z, a–z. Exclude symbol/icon fonts (those failing the 52-glyph check mostly drop out naturally).
- Render each glyph to a 64×64 grayscale bitmap with a **per-font consistent em scale and fixed baseline** (not per-glyph bounding-box fitting), so relative sizes, x-height and descenders are preserved. This matters because generated glyphs are assembled into a font with shared metrics.
- Split **by family** (not by style) into train / validation / test (~90/5/5) to avoid leakage between e.g. Roboto Regular and Roboto Bold.
- Store as compressed uint8 arrays plus a manifest (font name, family, licence).

### Matching the mouse-drawn input
The model learns from clean professional glyphs; visitors draw shaky mouse strokes. Mitigations:
- Reference-glyph augmentation in training: random stroke-thickness changes (erode/dilate), mild elastic distortion, small rotation/jitter.
- The drawing canvas shows baseline, x-height and cap-height guide lines, and drawings are normalised with the same em-scale convention as the training data.
- A second input mode: upload a font file the visitor already has; its glyphs are rendered and used as references. This also gives a clean demo path.

### Model
- **Style encoder:** small CNN over k reference glyphs (k sampled 3–8 during training), each paired with its letter identity; outputs are mean-pooled into one style vector.
- **Generator:** U-Net (~10–25M params) on 64×64, conditioned on target-letter embedding (52 classes) + style vector (e.g. adaptive group-norm/FiLM) + timestep.
- **Objective:** flow matching (rectified flow). Sampling with Euler, 8–16 steps, so browser generation stays fast.
- Condition dropout (~10%) to allow classifier-free guidance if it helps; optional.
- **Reference letters at inference:** the UI asks for a fixed set of 5 letters chosen to expose style features, proposed: `H` (stems/serifs), `O` (curves/contrast), `n`, `a`, `g` (lowercase structure, descender). Training always includes this fixed set as one of the k-sampling modes so the model sees it.

### Training
- PyTorch, mixed precision, on Kaggle. Checkpoint every ~15–30 min to persistent output and save a sample grid (held-out fonts) each checkpoint to watch progress.
- Target: start the first real run by Saturday midday; it runs overnight.

### Export and in-browser inference
- Export the style encoder and one generator step as separate ONNX graphs; the Euler loop runs in JavaScript.
- ONNX Runtime Web with WebGPU execution provider and WASM fallback. fp16 or int8 weights if needed to meet the size budget.
- Generate all 52 glyphs as one batch.

### Tracing and font building (browser)
- Upsample/smooth each 64×64 glyph, threshold, and trace to outlines with a potrace-family tracer.
- Assemble with opentype.js: units-per-em 1000, fixed ascender/descender matching the rendering convention, advance width from glyph bounds plus side bearings, a space glyph. Download as `.otf`. No kerning.

### Frontend
- React + Vite (TypeScript), static build on GitHub Pages.
- Flow: draw the 5 reference letters (or upload a font) → Generate → grid of 52 glyphs with per-glyph regenerate → type-test box rendering a sentence in the generated font → Download.

### Retrieval baseline / placeholder
- Nearest training font to the references by distance between normalised glyph bitmaps of the same letters; output that font's glyphs.
- In the browser, it runs over a small bundled subset (~200 fonts, 1-bit packed, ~5 MB) as the Saturday placeholder. For evaluation it runs in Python over the full training set.

### Evaluation (Python, held-out test fonts)
- Given the 5 reference letters from a test font, generate the other 47; compare against the font's true glyphs.
- **Legibility:** a small 52-way glyph classifier trained on training fonts (more reliable than off-the-shelf OCR on single characters). Report accuracy on real test glyphs (ceiling), model output and baseline output.
- **Style consistency:** pixel-level similarity to the true glyphs (L1 and SSIM). Report model vs. baseline.
- **Real-input check:** a small set of genuinely mouse-drawn reference sets, judged qualitatively, with failures shown in the README.
- The legibility classifier is a different method from OCR. The README states exactly what was measured, and the resume wording is adjusted afterwards to match.

### Build order
1. **Sat morning:** data pipeline (render, split) on Kaggle → start training ASAP.
2. **Sat while training runs:** frontend with the retrieval placeholder, tracing, opentype.js export: the full flow working end to end.
3. **Sat night:** training runs overnight.
4. **Sun:** ONNX export, swap in the model, evaluation, README with results and failures, deploy.
5. **Fallback:** if training fails, ship with the placeholder and document what went wrong and what was tried.

### Repo layout (proposed)
`data/` rendering and splits · `train/` model and training · `export/` ONNX · `eval/` metrics and classifier · `web/` React app · `docs/`

## Who builds it
Claude writes the code; the author runs training on Kaggle, reviews everything, and owns the result. Work runs in parallel where possible (web app while the model trains). The plan **must include walkthrough checkpoints** after each major component (data pipeline, model and flow-matching loss, training loop, ONNX export and the JS sampling loop, tracing and font build, evaluation). At each one the author explains the component back without notes. The resume claims depend on the author being able to defend every part in an interview.

## Calls made during review (Claude's, correctable)
- **Drawing scale:** the canvas guide lines define the em box, mapped to the same em-scale/baseline convention as rendering.
- **Font styles:** keep all weights and italics from each family. They are distinct styles, and the family-level split prevents leakage.
- **Drawn letters:** the 5 reference letters go into the output font as the model's regenerated versions, not the raw drawings, so the whole font is stylistically consistent. The raw drawing is a later toggle if time allows.
- **Data on Kaggle:** clone `google/fonts` inside the Kaggle notebook (internet enabled) and render there. Publish the rendered arrays as a private Kaggle dataset so retries skip re-rendering.
- **Model size budget:** fp16 export first; int8 only if still over ~30 MB.

## Left to the builder
Exact U-Net widths/depths and parameter count within the budget; training hyperparameters; augmentation strengths; choice of tracing library; guidance scale; the exact baseline font subset; visual design of the web UI.
