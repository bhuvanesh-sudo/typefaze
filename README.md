# TypeFaze

**Draw five letters. Get a whole font.**

TypeFaze is a few-shot font generator. You draw a handful of letters in the browser, a small generative model trained from scratch fills in the rest of the alphabet in your style, and you download an installable `.otf` font. All inference is meant to run on your own device: no server.

> **Status:** model trained, web app built, evaluation done. Not yet tested by real users drawing with a mouse, and the live demo isn't deployed yet.

---

## How it works

```
 your 5 letters ──► style encoder ──► style vector ─┐
  (H O n a g)                                        │
                                                     ▼
 random noise ──► [ U-Net × 4 Euler steps ] ───► 52 glyph images (64×64)
                    conditioned on: time t,          │
                    which letter, your style         ▼
                                              trace to outlines
                                                     │
                                                     ▼
                                           TypeFaze-yourname.otf
```

- **Model:** a style encoder (a CNN over the reference letters) plus a conditional U-Net, 13.5M parameters in total. It's a 27 MB download in the browser (weights stored as fp16).
- **Objective:** flow matching (rectified flow). The model learns the straight-line direction from noise to a real glyph, x₁ − x₀. Generation starts from noise and takes Euler steps along that direction. Because the learned paths are nearly straight, 4 steps look the same as 12 ([comparison](docs/steps-12-6-4.png)), so the site uses 4.
- **Style conditioning:** the style vector modulates every U-Net block via adaptive GroupNorm. 10% of training steps hide the style, so classifier-free guidance can strengthen it at sampling time. A guidance strength of 2.0 was chosen on validation fonts ([tuning](docs/cfg-tuning-val.txt)).
- **In the browser:** ONNX Runtime Web runs the encoder once, then calls the exported one-step U-Net inside a JavaScript Euler loop. It uses WebGPU when available and falls back to WASM on the CPU. Glyphs are traced to outlines with a small custom tracer and assembled into an `.otf` with opentype.js. Nothing is sent to a server.
- **Drawn vs. designed letters:** the reference letters are randomly thickened, thinned, rotated and warped during training, so the model tolerates wobbly mouse-drawn input rather than only clean font outlines.

## Data

- **Source:** the [`google/fonts`](https://github.com/google/fonts) repository, `ofl/` directory only (SIL Open Font License). Variable fonts are rendered at their default instance.
- **Rendering:** every font is drawn at one shared convention (em = 48 px, baseline at row 50 of a 64×64 box, centred on the advance width). Relative proportions (x-height, descenders, letter widths) are preserved, so generated glyphs can be assembled into a real font.
- **Filtering:** fonts missing any of A–Z/a–z, with more than 2% of their ink outside the box, or that fail to render are skipped.
- **Current build:** 3,518 fonts kept, 236 skipped.
- **Split by family, not by file:** 3,194 train / 193 val / 131 test. Roboto Regular and Roboto Bold always land in the same split, so the test set contains genuinely unseen designs.

## Repository layout

```
web/               React + Vite app (drawing pad, ONNX sampler, tracer, font builder)
typefaze/          Python package
  glyphs.py        character set and rendering constants
  render.py        font file -> 52 glyph bitmaps
  data.py          dataset build, family split, training Dataset
  augment.py       "make it look hand-drawn" augmentation
  model.py         StyleEncoder, GlyphUNet
  flow.py          flow-matching loss, Euler sampler, checkpoints
  baseline.py      nearest-existing-font baseline
  classifier.py    52-way letter classifier used to score legibility
  metrics.py       L1, SSIM, sibling-family grouping
scripts/
  build_dataset.py render google/fonts into glyphs.npz + manifest.json
  train.py         training with checkpoints and sample grids
  export_onnx.py   checkpoint -> browser model files
  make_bank.py     font subset for the in-browser baseline
  evaluate.py      held-out evaluation -> docs/results.md
kaggle/README.md   step-by-step Kaggle notebooks (dataset build, training)
tests/             pytest suite
docs/              design spec and implementation plan
```

## Running it

**Requirements:** Python 3.10+, and a GPU for training (Kaggle's free T4 works).

```bash
pip install -e ".[dev]"
pytest -m "not slow"          # quick tests
pytest -m slow                # overfit-one-font check (several minutes on CPU)
```

**Build the dataset** (about 2 minutes once the fonts are downloaded):
```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/google/fonts
cd fonts && git sparse-checkout set ofl && cd ..
python scripts/build_dataset.py --ofl fonts/ofl --out data/typefaze-data
```

**Train:**
```bash
python scripts/train.py --data data/typefaze-data --out runs/run1 --max-hours 11
# resume after an interruption
python scripts/train.py --data data/typefaze-data --out runs/run1 --resume runs/run1/ckpt_latest.pt
```
Training saves `ckpt_latest.pt` (for resuming), `ckpt_ema.pt` (smoothed weights for inference) and a `samples_<step>.png` grid every 20 minutes. Each grid row shows a validation font's 5 reference letters followed by the generated alphabet.

The exact cells for Kaggle are in [`kaggle/README.md`](kaggle/README.md). The trained run took 11 hours on a free Kaggle T4 (40,705 steps, final loss about 0.029).

**Export for the browser, evaluate, and run the site:**
```bash
python scripts/export_onnx.py --ckpt runs/run1/ckpt_ema.pt --out web/public/models --fp16 --steps 4 --wasm-steps 4 --cfg 2.0
python scripts/evaluate.py --data data/typefaze-data --ckpt runs/run1/ckpt_ema.pt --out docs/results.md --steps 4 --cfg 2.0
cd web && npm install && npm run dev
```

## Roadmap

- [x] Glyph rendering with a shared em/baseline convention
- [x] Dataset build with a family-level split (3,518 fonts)
- [x] Style encoder + conditional U-Net, flow-matching training
- [x] Full training run (11 h on a Kaggle T4)
- [x] Web app: drawing pad, font upload, glyph grid with per-letter regenerate, type tester
- [x] Tracing glyphs to outlines and building an installable `.otf` in the browser
- [x] ONNX export and in-browser inference (WebGPU, with WASM fallback)
- [x] Evaluation on 131 held-out fonts against a nearest-font baseline
- [ ] Test with real mouse-drawn letters
- [ ] Live demo on GitHub Pages

## Results

Evaluated on the **131 held-out test fonts**, from families never seen in training. The model sees only **H O n a g** from each font and generates the other **47** letters (4 steps, guidance 2.0, exactly the browser's settings). Full report: [`docs/results.md`](docs/results.md).

| | Legibility (classifier accuracy) | Mean L1 ↓ | Mean SSIM ↑ |
|---|---|---|---|
| Real letters (ceiling) | 91.5% | 0.000 | 1.000 |
| **TypeFaze** | **90.8%** | 0.031 | 0.867 |
| Nearest existing font (baseline) | 91.2% | 0.028 | 0.884 |

**What this says, honestly:**

- **TypeFaze's letters are about as legible as real ones.** A separate letter classifier reads them at 90.8%, against 91.5% for the real fonts. The gaps between the three rows (under one point) are within run-to-run noise: the classifier isn't seeded, and the "Real" row itself moved between 92.0% and 91.5% across two runs.
- **On Google Fonts, TypeFaze does not beat simply picking the closest existing font.** The baseline matches the hidden font's real letters a little better (SSIM 0.884 vs 0.867). Google Fonts contains many near-identical sans-serifs, so for a typical test font a very close real design already exists, and returning its crisp, real letters is hard to beat at the pixel level.
- **This isn't data leakage.** Only 12 of the 131 test fonts have a sibling family in training (e.g. a Condensed cut), and the baseline wins in both groups.
- **Where generation should matter is input that no existing font matches**, like your own drawing. That's untested so far; mouse-drawn tests are the next step.
- **Guidance helped measurably.** On 30 validation fonts at a fixed 4 steps, SSIM rose from 0.847 with no guidance to 0.879 at strength 2.0 ([tuning](docs/cfg-tuning-val.txt)).

The worst cases ([gallery](docs/failures.png)) are mostly fonts that aren't really alphabets: a dingbat font (Yarndings), letters drawn inside keyboard keys (Libertinus Keyboard), and pixel or textured display faces.

## Known limitations

- **64×64 glyphs:** fine detail and very thin (light-weight) strokes are hard to reproduce at this resolution.
- **Latin A–Z and a–z only:** no digits, punctuation or other scripts yet.
- **Dataset skew:** Google Fonts is dominated by sans-serifs, so expect the model to be strongest there and weaker on scripts and decorative styles.
- **Mouse-drawn input differs from professional fonts.** Augmentation helps, but this gap is the main expected failure mode, and it's untested so far.
- **Slow without WebGPU.** The CPU fallback runs single-threaded (GitHub Pages can't enable WASM threads). Measured with Node WASM on a 2-core cloud machine at the shipped settings: 24.7 s for 8 letters, so about 2½ minutes for all 52. Real laptops will vary. The WebGPU path hasn't been timed yet.
- **Non-alphabet fonts in the data.** Dingbat and symbol fonts that contain A–Z codepoints weren't filtered out of training or test.

## Licence

Training data: fonts from `google/fonts` under the SIL Open Font License 1.1. The test fixture `tests/fixtures/Lato-Regular.ttf` is OFL, with its licence alongside it.
