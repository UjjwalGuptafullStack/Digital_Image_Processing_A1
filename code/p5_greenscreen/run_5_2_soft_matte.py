"""5.2 Soft matte and spill suppression:
(a) chroma-distance distributions for backing / partial-alpha / foreground
(b) continuous alpha, composite, fraction of fractional-alpha pixels
(c) spill suppression, residual green excess over fractional-alpha pixels
(d) naive vs soft matte comparison (MAD of alpha maps and gradient fields)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from starter import (to_ycbcr, estimate_key_colour, key_soft, key_naive_rgb,
                      suppress_spill, composite, grad_error)

ROOT = Path(__file__).resolve().parents[2]
IMG_OUT = ROOT / "Report" / "Images"
IMG_OUT.mkdir(parents=True, exist_ok=True)
FRAMES = ROOT / "images/p5/frames"

frame = np.asarray(Image.open(FRAMES / "f024.png"))
H, W, _ = frame.shape
key_cbcr = estimate_key_colour(frame)
print(f"estimated key (Cb,Cr) = {key_cbcr}")

ycc = to_ycbcr(frame)
Cb, Cr = ycc[..., 1], ycc[..., 2]
dist = np.sqrt((Cb - key_cbcr[0]) ** 2 + (Cr - key_cbcr[1]) ** 2)

# ============================================================================
# (a) chroma distance distributions: backing / partial / foreground
# ============================================================================
backing_dist = dist[0:55, :].flatten()
face_dist = dist[220:340, 400:550].flatten()
jacket_dist = dist[450:600, 500:800].flatten()
fg_dist = np.concatenate([face_dist, jacket_dist])
hair_dist = dist[90:170, 370:530].flatten()

print(f"\nbacking:    n={len(backing_dist):6d}  mean={backing_dist.mean():6.2f}  "
      f"p1={np.percentile(backing_dist,1):5.2f}  p99={np.percentile(backing_dist,99):5.2f}")
print(f"foreground: n={len(fg_dist):6d}  mean={fg_dist.mean():6.2f}  "
      f"p1={np.percentile(fg_dist,1):5.2f}  p99={np.percentile(fg_dist,99):5.2f}")
print(f"hair-edge:  n={len(hair_dist):6d}  mean={hair_dist.mean():6.2f}  "
      f"p1={np.percentile(hair_dist,1):5.2f}  p99={np.percentile(hair_dist,99):5.2f}")

separates = np.percentile(backing_dist, 99) < np.percentile(fg_dist, 1)
print(f"\nbacking p99 ({np.percentile(backing_dist,99):.2f}) < "
      f"foreground p1 ({np.percentile(fg_dist,1):.2f})? {separates}")

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, 100, 80)
ax.hist(backing_dist, bins=bins, alpha=0.6, label="pure backing", color="green", density=True)
ax.hist(hair_dist, bins=bins, alpha=0.6, label="hair edge (partial)", color="orange", density=True)
ax.hist(fg_dist, bins=bins, alpha=0.6, label="pure foreground (face+jacket)", color="steelblue",
        density=True)
ax.set_xlabel("chroma distance from key $\\|(C_b,C_r)-\\mathrm{key}\\|$")
ax.set_ylabel("density")
ax.set_title("5.2(a) Chroma distance: backing / partial-alpha / foreground populations")
ax.legend()
fig.tight_layout()
fig.savefig(IMG_OUT / "p5_5_2_chroma_distributions.png", dpi=150)
plt.close(fig)

# thresholds chosen from the data above: t_in just above the backing p99,
# t_out at the foreground p1 (both populations' tails, giving margin)
T_IN = 12.0
T_OUT = 58.0
print(f"\nchosen t_in={T_IN} (backing p99={np.percentile(backing_dist,99):.1f}), "
      f"t_out={T_OUT} (foreground p1={np.percentile(fg_dist,1):.1f})")

# ============================================================================
# (b) continuous alpha + composite
# ============================================================================
alpha = key_soft(frame, key_cbcr=key_cbcr, t_in=T_IN, t_out=T_OUT)
frac_mask = (alpha > 0) & (alpha < 1)
print(f"\nfraction of pixels with fractional alpha: {frac_mask.mean()*100:.2f}%")

bg_plate = np.asarray(Image.open(ROOT / "images/p5/pano.jpg"))
bg_h, bg_w, _ = bg_plate.shape
# static crop for this single-frame demo (panning handled in 5.3)
bg_crop = bg_plate[:H, :W] if bg_h >= H and bg_w >= W else np.array(
    Image.fromarray(bg_plate).resize((W, H)))

composite_nospill = composite(frame, bg_crop, alpha, spill=False)
composite_spill = composite(frame, bg_crop, alpha, spill=True, key_cbcr=key_cbcr,
                              spill_strength=3.0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
axes[0].imshow(frame)
axes[0].set_title("Original frame")
axes[1].imshow(alpha, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Soft alpha matte\n({frac_mask.mean()*100:.1f}% fractional)")
axes[2].imshow(composite_spill)
axes[2].set_title("Composite (with spill suppression)")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(IMG_OUT / "p5_5_2_soft_alpha_composite.png", dpi=150)
plt.close(fig)

# with vs. without spill suppression, zoomed on the hair edge where spill is visible
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
zy0, zy1, zx0, zx1 = 60, 300, 350, 750
axes[0].imshow(composite_nospill[zy0:zy1, zx0:zx1])
axes[0].set_title("Composite, no spill suppression")
axes[1].imshow(composite_spill[zy0:zy1, zx0:zx1])
axes[1].set_title("Composite, with spill suppression")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(IMG_OUT / "p5_5_2_composite_spill_compare.png", dpi=150)
plt.close(fig)

# ============================================================================
# (c) spill suppression: residual green excess over fractional-alpha pixels
# ============================================================================
def green_excess(rgb):
    rgb = rgb.astype(np.float64)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return np.maximum(G - np.maximum(R, B), 0.0)

excess_before = green_excess(frame)
excess_before_frac = excess_before[frac_mask]

# strength sweep: correction magnitude is strength * excess * (1-alpha), and
# at typical hair-edge alpha (~0.6) only ~40% of raw excess is removed at
# strength=1, so a sweep is needed to see how much strength is actually
# required to clear spill rather than assuming strength=1 suffices.
print("\nstrength sweep (residual green excess, fractional-alpha pixels only):")
SPILL_STRENGTH = 3.0
for s in (1.0, 2.0, SPILL_STRENGTH, 5.0):
    d = suppress_spill(frame, alpha, strength=s, key_cbcr=key_cbcr)
    e = green_excess(d)[frac_mask]
    print(f"  strength={s:.1f}: mean residual excess = {e.mean():6.2f} "
          f"({100*(1-e.mean()/excess_before_frac.mean()):.1f}% reduction)")

despilled = suppress_spill(frame, alpha, strength=SPILL_STRENGTH, key_cbcr=key_cbcr)
excess_after_frac = green_excess(despilled)[frac_mask]
print(f"\nchosen strength={SPILL_STRENGTH}: before={excess_before_frac.mean():.2f}  "
      f"after={excess_after_frac.mean():.2f}  "
      f"({100*(1-excess_after_frac.mean()/excess_before_frac.mean()):.1f}% reduction)")

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
crop_y0, crop_y1, crop_x0, crop_x1 = 60, 300, 350, 750
axes[0].imshow(frame[crop_y0:crop_y1, crop_x0:crop_x1])
axes[0].set_title("Before spill suppression (crop)")
axes[1].imshow(despilled[crop_y0:crop_y1, crop_x0:crop_x1])
axes[1].set_title("After spill suppression (crop)")
axes[2].hist(excess_before_frac, bins=40, alpha=0.6, label=f"before (mean={excess_before_frac.mean():.2f})",
             color="indianred", density=True)
axes[2].hist(excess_after_frac, bins=40, alpha=0.6, label=f"after (mean={excess_after_frac.mean():.2f})",
             color="steelblue", density=True)
axes[2].set_xlabel("green excess $\\max(G-\\max(R,B),0)$")
axes[2].set_title("Green excess, fractional-alpha pixels only")
axes[2].legend(fontsize=8)
for ax in axes[:2]:
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(IMG_OUT / "p5_5_2_spill_suppression.png", dpi=150)
plt.close(fig)

# ============================================================================
# (d) naive vs soft matte comparison
# ============================================================================
alpha_naive = key_naive_rgb(frame, thresh=60)
mad_alpha = float(np.mean(np.abs(alpha_naive - alpha)))
gradient_mad = grad_error(alpha_naive * 255, alpha * 255) * 1000 / alpha.size  # per-pixel scale
print(f"\nMAD(alpha_naive, alpha_soft) = {mad_alpha:.4f}")
print(f"grad_error(alpha_naive, alpha_soft) [raw sad-of-gradients/1000] = "
      f"{grad_error(alpha_naive*255, alpha*255):.2f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
axes[0].imshow(alpha_naive, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Naive (hard) matte")
axes[1].imshow(alpha, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Soft matte")
diff = np.abs(alpha_naive - alpha)
im = axes[2].imshow(diff, cmap="inferno", vmin=0, vmax=1)
axes[2].set_title(f"|naive - soft|\nMAD={mad_alpha:.3f}")
plt.colorbar(im, ax=axes[2], fraction=0.046)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(IMG_OUT / "p5_5_2_naive_vs_soft.png", dpi=150)
plt.close(fig)

print(f"\nSaved all 5.2 figures to {IMG_OUT}")
