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

## Notebook 2: training (Task 6)

Instructions will be added once the training script exists.
