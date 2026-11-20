# Running TypeFaze on Kaggle

Replace `YOUR_GITHUB_USERNAME` below with your GitHub username. The TypeFaze repo must be pushed and public.

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
!git clone https://github.com/YOUR_GITHUB_USERNAME/typefaze /kaggle/temp/typefaze
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
!git clone https://github.com/YOUR_GITHUB_USERNAME/typefaze /kaggle/temp/typefaze
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
