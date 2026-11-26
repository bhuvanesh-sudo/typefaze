# Running TypeFaze on Kaggle

## Notebook 1: build the dataset (Task 3, ~10–20 min, no GPU needed)

New notebook → **Settings: Internet ON, Accelerator: None**. Run these cells in order.

**Cell 1: get the fonts (OFL only)**
```bash
!git clone --depth 1 --filter=blob:none --sparse https://github.com/google/fonts /kaggle/temp/fonts
!cd /kaggle/temp/fonts && git sparse-checkout set ofl
!ls /kaggle/temp/fonts/ofl | wc -l
```
Expect roughly 1,500+ family folders.

**Cell 2: install TypeFaze**
```bash
!git clone https://github.com/bhuvanesh-sudo/typefaze /kaggle/temp/typefaze
!pip install -q -e /kaggle/temp/typefaze
```

**Cell 3: render**
```bash
!cd /kaggle/temp/typefaze && python scripts/build_dataset.py --ofl /kaggle/temp/fonts/ofl --out /kaggle/working/typefaze-data
```
Check the printed counts: `kept` should be at least 1,200, and `skipped` is reported. Paste the counts back to Claude.

**Cell 4: visual check (10 val fonts × 52 letters)**
```python
import json, numpy as np, matplotlib.pyplot as plt
d = "/kaggle/working/typefaze-data"
m = json.load(open(f"{d}/manifest.json"))
g = np.load(f"{d}/glyphs.npz")["glyphs"]
val = [i for i, f in enumerate(m["fonts"]) if f["split"] == "val"][:10]
rows = [np.concatenate(list(g[i]), axis=1) for i in val]
grid = np.concatenate(rows, axis=0).astype(float)
grid[50::64, :] = np.maximum(grid[50::64, :], 90)   # baseline guide
plt.figure(figsize=(40, 8)); plt.imshow(grid, cmap="gray"); plt.axis("off")
```
Check that the letters sit on the grey baseline and that nothing is cut off.

**Cell 5: publish.** Click **Save Version → Save & Run All**. When it finishes, open the notebook's **Output** tab → **New Dataset**, name it `typefaze-data`, and keep it private.

## Notebook 2: training (Task 6, runs ~11 h)

New notebook → **Settings: Internet ON, Accelerator: GPU T4 x1** (or P100). **Add Input** → your private `typefaze-data` dataset.

**Cell 1: install**
```bash
!git clone https://github.com/bhuvanesh-sudo/typefaze /kaggle/temp/typefaze
!pip install -q -e /kaggle/temp/typefaze
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

**Cell 2: train**
```bash
!cd /kaggle/temp/typefaze && python scripts/train.py --data /kaggle/input/typefaze-data --out /kaggle/working/run1 --max-hours 11 --ckpt-minutes 20
```
Within about 2 minutes you should see a `{"step": 100, "loss": ...}` line, and the loss should trend down. A sample grid appears every 20 minutes in `/kaggle/working/run1/samples_*.png`: each row shows a validation font's 5 reference letters, a grey gap, then the generated A–Z a–z. Early grids are blobs; that's normal.

To run overnight, use **Save Version → Save & Run All (commit)**. That runs in the background even when you close the tab, and keeps everything in `/kaggle/working` as the version's output.

**If the session dies,** start a new notebook with the previous version's output added as an input and run:
```bash
!cd /kaggle/temp/typefaze && python scripts/train.py --data /kaggle/input/typefaze-data --out /kaggle/working/run1 --max-hours 11 --resume /kaggle/input/<previous-output>/run1/ckpt_latest.pt
```

**When it finishes:** from the version's Output, create a private dataset `typefaze-ckpt` containing `ckpt_ema.pt` plus the last few `samples_*.png`.

---

# v2: element-carrying, cursive-leaning model

## Notebook 3: build the v2 dataset (~5 min, no GPU)

Same as Notebook 1, with two changes: the checkout also fetches each family's `METADATA.pb` (its Google Fonts category), and handwriting fonts get a looser clipping limit so script fonts with swashes are kept.

**Cell 1: fonts and metadata**
```bash
!rm -rf /kaggle/temp/fonts
!git clone --depth 1 --filter=blob:none --sparse https://github.com/google/fonts /kaggle/temp/fonts
!cd /kaggle/temp/fonts && git sparse-checkout set --no-cone '/ofl/*/*.ttf' '/ofl/*/*.otf' '/ofl/*/METADATA.pb'
```

**Cell 2: install TypeFaze**
```bash
!rm -rf /kaggle/temp/typefaze
!git clone https://github.com/bhuvanesh-sudo/typefaze /kaggle/temp/typefaze
!pip install -q -e /kaggle/temp/typefaze
```

**Cell 3: render**
```bash
!cd /kaggle/temp/typefaze && python scripts/build_dataset.py --ofl /kaggle/temp/fonts/ofl --out /kaggle/working/typefaze-data-v2 --handwriting-clip 0.06
```
Expected counts: about `kept` 3,588, `handwriting_kept` 310, and a split of about 3,258 / 196 / 134. Send them to Claude.

**Cell 4: publish.** Save & Run All, then turn the output into a private dataset named `typefaze-data-v2`.

## Notebook 4: train v2 (GPU, ~11 h per session)

Settings: GPU T4 x1 (or P100), Internet on. Add Input → `typefaze-data-v2`.

**Cell 1: install** (the same as Cell 2 above)

**Cell 2: train**
```bash
!cd /kaggle/temp/typefaze && python scripts/train.py --arch v2 --data /kaggle/input/typefaze-data-v2 --out /kaggle/working/run-v2 --max-hours 11 --ckpt-minutes 20 --workers 4
```
`--workers 4` matters: v2 draws its decorations on the fly, so it needs Kaggle's CPU cores to keep the GPU busy. If the `--data` path isn't found, run `!find /kaggle/input -name glyphs.npz` and point `--data` at the folder above it, as before.

**What to watch:**
- loss lines every 100 steps, trending down;
- `samples_*.png` every 20 minutes. Each row is labelled:
  - `plain`: no element;
  - a training element name;
  - a name ending in ` *`: an element the model **never trained on**. Those two rows are the interesting ones.

**Run it overnight:** Save Version → Save & Run All (Commit).

**Second session (optional, if the first session's grids are still improving):** add the first run's output as an input and append `--resume /kaggle/input/<previous-output>/run-v2/ckpt_latest.pt`.

**When done:** download `run-v2/ckpt_ema.pt` (and the last couple of sample grids) and put them in `C:\Projects\typefaze\runs\v2\`.
