"""4.1 Global equalisation: input.png before/after, histograms and CDFs,
plus the idempotence check and the near-no-op uniform-histogram example.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from starter import equalise, hist, cdf

ROOT = Path(__file__).resolve().parents[2]
IMG_OUT = ROOT / "Report" / "Images"
IMG_OUT.mkdir(parents=True, exist_ok=True)

img = np.asarray(Image.open(ROOT / "images/p4/equalization/input.png"))
eq = equalise(img)

h_before = hist(img)
h_after = hist(eq)
F_before = cdf(h_before)
F_after = cdf(h_after)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("Before")
axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
axes[0, 1].bar(range(256), h_before, width=1.0, color="steelblue")
axes[0, 1].set_title("Histogram before")
axes[0, 2].plot(F_before, color="steelblue")
axes[0, 2].set_title("CDF before")
axes[0, 2].set_ylim(0, 1.02)

axes[1, 0].imshow(eq, cmap="gray")
axes[1, 0].set_title("After equalisation")
axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
axes[1, 1].bar(range(256), h_after, width=1.0, color="indianred")
axes[1, 1].set_title("Histogram after")
axes[1, 2].plot(F_after, color="indianred")
axes[1, 2].set_title("CDF after")
axes[1, 2].set_ylim(0, 1.02)

for ax in (axes[0, 1], axes[1, 1]):
    ax.set_xlabel("level")
    ax.set_ylabel("pixel count")
for ax in (axes[0, 2], axes[1, 2]):
    ax.set_xlabel("level")
    ax.set_ylabel("F(v)")

fig.tight_layout()
fig.savefig(IMG_OUT / "p4_4_1_before_after.png", dpi=150)
plt.close(fig)

# --- idempotence check (numerical confirmation of the proof) -----------
once = equalise(img)
twice = equalise(once)
print("idempotence: max|equalise(equalise(I)) - equalise(I)| =",
      np.abs(once.astype(int) - twice.astype(int)).max())
print("identical:", np.array_equal(once, twice))

# --- (c) near-no-op example: exactly uniform histogram -------------------
rng = np.random.default_rng(0)
side = 256
counts_per_level = (side * side) // 256
flat = np.repeat(np.arange(256, dtype=np.uint8), counts_per_level)
rng.shuffle(flat)
uniform_img = flat.reshape(side, side)
uniform_eq = equalise(uniform_img)
diff = np.abs(uniform_img.astype(int) - uniform_eq.astype(int))
print(f"\nuniform-histogram image: max|diff|={diff.max()} mean|diff|={diff.mean():.4f} "
      f"(pure rounding, no perceptible change)")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
axes[0].imshow(uniform_img, cmap="gray")
axes[0].set_title("Constructed uniform-histogram image")
axes[1].bar(range(256), hist(uniform_img), width=1.0, color="steelblue")
axes[1].set_title("Histogram (exactly uniform)")
axes[1].set_xlabel("level"); axes[1].set_ylabel("pixel count")
axes[2].imshow(np.abs(uniform_img.astype(int) - uniform_eq.astype(int)).astype(np.uint8),
               cmap="gray", vmin=0, vmax=1)
axes[2].set_title(f"|equalise(I) - I| (max={diff.max()}, mean={diff.mean():.3f})")
axes[0].set_xticks([]); axes[0].set_yticks([])
axes[2].set_xticks([]); axes[2].set_yticks([])
fig.tight_layout()
fig.savefig(IMG_OUT / "p4_4_1_near_noop.png", dpi=150)
plt.close(fig)

print(f"\nSaved before/after and near-no-op figures to {IMG_OUT}")
