"""1.1 Correctness: run all four conv2d implementations on the standard test
case (128x128 crop of base_2048.png, k=7 Gaussian), verify against
scipy.signal.convolve2d, and save the per-method |error| heatmap into
Report/Images/ for the report and the viva.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d  # checking only

sys.path.insert(0, str(Path(__file__).resolve().parent))
from starter import kernel_bank, conv2d_loops, conv2d_taps, conv2d_im2col, conv2d_fft

ROOT = Path(__file__).resolve().parents[2]
IMG_OUT = ROOT / "Report" / "Images"
IMG_OUT.mkdir(parents=True, exist_ok=True)

img = np.asarray(Image.open(ROOT / "images/p1/base_2048.png")).astype(float)[:128, :128]
K = kernel_bank(7)["gaussian"]
ref = convolve2d(img, K, mode="same", boundary="fill")

methods = {
    "conv2d_loops": conv2d_loops,
    "conv2d_taps": conv2d_taps,
    "conv2d_im2col": conv2d_im2col,
    "conv2d_fft": conv2d_fft,
}

outputs = {}
errors = {}
for name, fn in methods.items():
    out = fn(img, K)
    outputs[name] = out
    err = np.abs(out - ref)
    errors[name] = err
    print(f"{name:16s} max|err| = {err.max():.3e}")

# Per-method |error| heatmaps, shared colour scale
max_err = max(e.max() for e in errors.values())
fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
for ax, (name, err) in zip(axes, errors.items()):
    im = ax.imshow(err, cmap="inferno", vmin=0, vmax=max(max_err, 1e-12))
    ax.set_title(f"{name}\nmax|err|={err.max():.2e}")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Absolute error vs. scipy.signal.convolve2d (128x128 crop, k=7 Gaussian)")
fig.tight_layout()
fig.savefig(IMG_OUT / "p1_1_1_error_heatmaps.png", dpi=150)
plt.close(fig)

print(f"\nSaved error heatmaps to {IMG_OUT}")
