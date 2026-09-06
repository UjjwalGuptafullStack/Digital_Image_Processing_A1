"""4.3 Local histogram processing: patch histograms, plain AHE (its two
failure modes), CLAHE, clip/tile ablation grid, and local specification.
Uses local_regions.png converted once to BT.601 luminance.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from starter import hist, cdf, ahe, clahe, specify, _tile_bounds

ROOT = Path(__file__).resolve().parents[2]
IMG_OUT = ROOT / "Report" / "Images"
IMG_OUT.mkdir(parents=True, exist_ok=True)

rgb = np.asarray(Image.open(ROOT / "images/p4/local_regions.png"))
Y = np.round(0.299 * rgb[..., 0].astype(np.float64) +
             0.587 * rgb[..., 1].astype(np.float64) +
             0.114 * rgb[..., 2].astype(np.float64)).astype(np.uint8)
H, W = Y.shape
print(f"Y shape: {Y.shape}")

# ============================================================================
# (a) five distinct local patches, histograms
# ============================================================================
patch_size = 120
rng = np.random.default_rng(3)
# hand-pick 5 patches spanning a bright, a dark, and mixed regions by
# sampling candidate patches and keeping ones with distinct mean brightness
candidates = []
for _ in range(200):
    y0 = rng.integers(0, H - patch_size)
    x0 = rng.integers(0, W - patch_size)
    patch = Y[y0:y0 + patch_size, x0:x0 + patch_size]
    candidates.append((y0, x0, patch.mean(), patch.std()))
candidates.sort(key=lambda c: c[2])
picks_idx = np.linspace(0, len(candidates) - 1, 5).astype(int)
picks = [candidates[i] for i in picks_idx]

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
for col, (y0, x0, mean_, std_) in enumerate(picks):
    patch = Y[y0:y0 + patch_size, x0:x0 + patch_size]
    axes[0, col].imshow(patch, cmap="gray", vmin=0, vmax=255)
    axes[0, col].set_title(f"Patch {col+1}\nmean={mean_:.0f} std={std_:.0f}")
    axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
    axes[1, col].bar(range(256), hist(patch), width=1.0, color="steelblue")
    axes[1, col].set_xlim(0, 255)
    axes[1, col].set_xlabel("level")
fig.suptitle("4.3(a) Five distinct local patches: no single global mapping serves all")
fig.tight_layout()
fig.savefig(IMG_OUT / "p4_4_3_patches.png", dpi=150)
plt.close(fig)
print("(a) patch means:", [f"{c[2]:.1f}" for c in picks])

# ============================================================================
# (b) plain tiled AHE -- show its two failure modes
# ============================================================================
ahe_out = ahe(Y, tiles=8)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
axes[0].imshow(Y, cmap="gray")
axes[0].set_title("Original luminance")
axes[1].imshow(ahe_out, cmap="gray")
axes[1].set_title("Plain tiled AHE (tiles=8): block edges + noise amplification")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(IMG_OUT / "p4_4_3_ahe_failure.png", dpi=150)
plt.close(fig)

# zoom on a flat/near-uniform tile to show noise amplification specifically
ys = _tile_bounds(H, 8)
xs = _tile_bounds(W, 8)
tile_stds = np.zeros((8, 8))
for i in range(8):
    for j in range(8):
        tile_stds[i, j] = Y[ys[i]:ys[i+1], xs[j]:xs[j+1]].std()
flat_i, flat_j = np.unravel_index(np.argmin(tile_stds), tile_stds.shape)
y0, y1, x0, x1 = ys[flat_i], ys[flat_i+1], xs[flat_j], xs[flat_j+1]
print(f"(b) flattest tile: ({flat_i},{flat_j}), std={tile_stds[flat_i,flat_j]:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(Y[y0:y1, x0:x1], cmap="gray")
axes[0].set_title(f"Original (flattest tile, std={tile_stds[flat_i,flat_j]:.2f})")
axes[1].imshow(ahe_out[y0:y1, x0:x1], cmap="gray")
axes[1].set_title("Same tile after AHE: noise amplified")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(IMG_OUT / "p4_4_3_ahe_noise_zoom.png", dpi=150)
plt.close(fig)

# ============================================================================
# (c) CLAHE
# ============================================================================
clahe_out = clahe(Y, tiles=8, clip=3.0)

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
axes[0].imshow(Y, cmap="gray")
axes[0].set_title("Original")
axes[1].imshow(ahe_out, cmap="gray")
axes[1].set_title("Plain AHE")
axes[2].imshow(clahe_out, cmap="gray")
axes[2].set_title("CLAHE (tiles=8, clip=3.0)")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(IMG_OUT / "p4_4_3_clahe.png", dpi=150)
plt.close(fig)

# ============================================================================
# (d) ablation grid: clip x tiles
# ============================================================================
clip_values = [1, 2, 3, 10, np.inf]
tile_values = [2, 4, 8, 16, 64]

fig, axes = plt.subplots(len(clip_values), len(tile_values), figsize=(18, 18))
for ci, clip in enumerate(clip_values):
    for ti, tiles in enumerate(tile_values):
        out = clahe(Y, tiles=tiles, clip=clip)
        axes[ci, ti].imshow(out, cmap="gray")
        axes[ci, ti].set_xticks([]); axes[ci, ti].set_yticks([])
        clip_lbl = "$\\infty$" if np.isinf(clip) else str(clip)
        axes[ci, ti].set_title(f"clip={clip_lbl}, tiles={tiles}", fontsize=9)
fig.suptitle("4.3(d) CLAHE ablation: clip limit x tile count")
fig.tight_layout()
fig.savefig(IMG_OUT / "p4_4_3_ablation.png", dpi=120)
plt.close(fig)
print("(d) ablation grid saved")

# ============================================================================
# (e) local histogram specification vs CLAHE
# ============================================================================
# Pick a region with a strongly bimodal/skewed histogram (bright sky +
# dark building silhouette) where a UNIFORM target (what CLAHE effectively
# aims for) crushes one mode together, but specifying a bimodal target
# preserves the separation between the two modes.
tile_variances = np.zeros((8, 8))
tile_bimodality = np.zeros((8, 8))
ys8 = _tile_bounds(H, 8)
xs8 = _tile_bounds(W, 8)
for i in range(8):
    for j in range(8):
        t = Y[ys8[i]:ys8[i+1], xs8[j]:xs8[j+1]]
        h = hist(t)
        # crude bimodality proxy: two well-separated populous halves
        lo_mass = h[:100].sum()
        hi_mass = h[156:].sum()
        tile_bimodality[i, j] = min(lo_mass, hi_mass)
bi, bj = np.unravel_index(np.argmax(tile_bimodality), tile_bimodality.shape)
y0, y1, x0, x1 = ys8[bi], ys8[bi+1], xs8[bj], xs8[bj+1]
region = Y[y0:y1, x0:x1]
print(f"(e) chosen region ({bi},{bj}) for bimodal local specification demo")

region_hist = hist(region)
# Bimodal target: sum of two Gaussians centred on the region's own two modes
# (this is the "extra information" the method needs: an appropriate TARGET
# shape, here estimated from the region's own histogram peaks rather than
# assumed uniform).
smoothed = np.convolve(region_hist, np.ones(9)/9, mode="same")
half = len(smoothed)//2
peak_lo = np.argmax(smoothed[:half])
peak_hi = half + np.argmax(smoothed[half:])
levels = np.arange(256)
target_bimodal = (np.exp(-0.5*((levels-peak_lo)/15)**2) +
                   np.exp(-0.5*((levels-peak_hi)/15)**2))
target_bimodal = target_bimodal / target_bimodal.sum() * region.size

region_clahe = clahe(region, tiles=1, clip=3.0)  # tiles=1: global-equalise-like within region
region_specified = specify(region, target_bimodal)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes[0, 0].imshow(region, cmap="gray")
axes[0, 0].set_title("Original region")
axes[0, 1].imshow(region_clahe, cmap="gray")
axes[0, 1].set_title("CLAHE-style (uniform target)")
axes[0, 2].imshow(region_specified, cmap="gray")
axes[0, 2].set_title("Local specification (bimodal target)")
for ax in axes[0]:
    ax.set_xticks([]); ax.set_yticks([])

axes[1, 0].bar(range(256), region_hist, width=1.0, color="steelblue")
axes[1, 0].set_title("Original histogram (bimodal)")
axes[1, 1].bar(range(256), hist(region_clahe), width=1.0, color="indianred")
axes[1, 1].set_title("CLAHE result histogram")
axes[1, 2].bar(range(256), hist(region_specified), width=1.0, color="darkgreen", alpha=0.7, label="achieved")
axes[1, 2].plot(target_bimodal, color="black", label="target")
axes[1, 2].set_title("Specified result vs. bimodal target")
axes[1, 2].legend(fontsize=8)
fig.suptitle("4.3(e) Local specification (peaks estimated from the region) vs. CLAHE")
fig.tight_layout()
fig.savefig(IMG_OUT / "p4_4_3_local_specification.png", dpi=150)
plt.close(fig)

print(f"\nSaved all 4.3 figures to {IMG_OUT}")
