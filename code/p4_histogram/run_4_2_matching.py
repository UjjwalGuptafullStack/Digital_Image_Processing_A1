"""4.2 Histogram matching and specification:
(a) match matching/source.png to matching/reference.png's histogram
(b) specify uniform and Gaussian(128,35) targets on specification/source.png
(c) colour: per-channel RGB vs luminance-only equalisation on colour_source.png
Reports W1 distance (achieved vs target) before/after for all three targets.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from starter import specify, wasserstein1, hist, cdf, equalise, apply_on_luma

ROOT = Path(__file__).resolve().parents[2]
IMG_OUT = ROOT / "Report" / "Images"
IMG_OUT.mkdir(parents=True, exist_ok=True)

w1_rows = []

# ============================================================================
# (a) Image matching
# ============================================================================
src = np.asarray(Image.open(ROOT / "images/p4/matching/source.png"))
ref = np.asarray(Image.open(ROOT / "images/p4/matching/reference.png"))

h_ref = hist(ref)
matched = specify(src, h_ref)

h_src = hist(src)
h_matched = hist(matched)
w1_before = wasserstein1(h_src, h_ref)
w1_after = wasserstein1(h_matched, h_ref)
w1_rows.append(("Image matching (source->reference)", w1_before, w1_after))
print(f"(a) W1(source,reference)={w1_before:.4f}  W1(matched,reference)={w1_after:.4f}")

fig, axes = plt.subplots(3, 3, figsize=(14, 11))
for col, (name, im) in enumerate([("Source", src), ("Reference", ref), ("Matched", matched)]):
    axes[0, col].imshow(im, cmap="gray")
    axes[0, col].set_title(name)
    axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
    h = hist(im)
    axes[1, col].bar(range(256), h, width=1.0, color="steelblue")
    axes[1, col].set_title(f"{name} histogram")
    F = cdf(h)
    axes[2, col].plot(F, color="indianred")
    axes[2, col].set_title(f"{name} CDF")
    axes[2, col].set_ylim(0, 1.02)
fig.suptitle(f"4.2(a) Image matching: W1 before={w1_before:.3f}, after={w1_after:.3f}")
fig.tight_layout()
fig.savefig(IMG_OUT / "p4_4_2_matching.png", dpi=150)
plt.close(fig)

# ============================================================================
# (b) Histogram specification: uniform and Gaussian targets
# ============================================================================
spec_src = np.asarray(Image.open(ROOT / "images/p4/specification/source.png"))
h_spec_src = hist(spec_src)

uniform_target = np.full(256, spec_src.size / 256.0)

levels = np.arange(256)
mu, sigma = 128.0, 35.0
gauss_target = np.exp(-0.5 * ((levels - mu) / sigma) ** 2)
gauss_target /= gauss_target.sum()
gauss_target *= spec_src.size  # match total pixel count for a fair W1/hist comparison

results_b = {}
for tname, target in (("uniform", uniform_target), ("gaussian", gauss_target)):
    out = specify(spec_src, target)
    h_out = hist(out)
    w1_b = wasserstein1(h_spec_src, target)
    w1_a = wasserstein1(h_out, target)
    results_b[tname] = (out, h_out, w1_b, w1_a)
    w1_rows.append((f"Specification ({tname} target)", w1_b, w1_a))
    print(f"(b) {tname}: W1(source,target)={w1_b:.4f}  W1(result,target)={w1_a:.4f}")

fig, axes = plt.subplots(3, 3, figsize=(14, 11))
axes[0, 0].imshow(spec_src, cmap="gray")
axes[0, 0].set_title("Source")
axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
axes[1, 0].bar(range(256), h_spec_src, width=1.0, color="steelblue")
axes[1, 0].set_title("Source histogram")
axes[2, 0].axis("off")

for col, tname in enumerate(("uniform", "gaussian"), start=1):
    out, h_out, w1_b, w1_a = results_b[tname]
    axes[0, col].imshow(out, cmap="gray")
    axes[0, col].set_title(f"Result ({tname} target)")
    axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
    axes[1, col].bar(range(256), h_out, width=1.0, color="indianred", alpha=0.7, label="achieved")
    target = uniform_target if tname == "uniform" else gauss_target
    axes[1, col].plot(target, color="black", linewidth=1.2, label="target")
    axes[1, col].set_title(f"{tname.capitalize()}: achieved vs target\nW1 before={w1_b:.2f}, after={w1_a:.2f}")
    axes[1, col].legend(fontsize=7)
    axes[2, col].plot(cdf(h_out), color="indianred", label="achieved CDF")
    axes[2, col].plot(cdf(target), color="black", linestyle="--", label="target CDF")
    axes[2, col].legend(fontsize=7)
    axes[2, col].set_ylim(0, 1.02)
fig.suptitle("4.2(b) Histogram specification: uniform and Gaussian($\\mu$=128,$\\sigma$=35) targets")
fig.tight_layout()
fig.savefig(IMG_OUT / "p4_4_2_specification.png", dpi=150)
plt.close(fig)

# ============================================================================
# (c) Colour: per-channel vs luminance-only equalisation
# ============================================================================
colour_src = np.asarray(Image.open(ROOT / "images/p4/colour_source.png"))


def rgb_to_hsv_np(rgb):
    """From-scratch RGB->HSV (H in [0,360), S,V in [0,1])."""
    rgb = rgb.astype(np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin
    h = np.zeros_like(cmax)
    mask = delta > 1e-12
    r_max = mask & (cmax == r)
    g_max = mask & (cmax == g) & ~r_max
    b_max = mask & (cmax == b) & ~r_max & ~g_max
    h[r_max] = 60.0 * (((g[r_max] - b[r_max]) / delta[r_max]) % 6)
    h[g_max] = 60.0 * (((b[g_max] - r[g_max]) / delta[g_max]) + 2)
    h[b_max] = 60.0 * (((r[b_max] - g[b_max]) / delta[b_max]) + 4)
    s = np.where(cmax > 1e-12, delta / np.maximum(cmax, 1e-12), 0.0)
    v = cmax
    return h, s, v


def circular_hue_distance(h1, h2):
    d = np.abs(h1 - h2) % 360.0
    return np.minimum(d, 360.0 - d)


SAT_THRESHOLD = 0.08  # exclude near-achromatic pixels (hue undefined/unstable)

per_channel = np.stack([equalise(colour_src[..., c]) for c in range(3)], axis=-1)
luma_only = apply_on_luma(colour_src, equalise)

h_orig, s_orig, v_orig = rgb_to_hsv_np(colour_src)
h_pc, s_pc, v_pc = rgb_to_hsv_np(per_channel)
h_luma, s_luma, v_luma = rgb_to_hsv_np(luma_only)

chromatic_mask = s_orig > SAT_THRESHOLD
d_pc = circular_hue_distance(h_orig[chromatic_mask], h_pc[chromatic_mask])
d_luma = circular_hue_distance(h_orig[chromatic_mask], h_luma[chromatic_mask])
print(f"\n(c) mean hue shift, per-channel equalisation: {d_pc.mean():.2f} deg "
      f"(saturation threshold {SAT_THRESHOLD}, {chromatic_mask.sum()}/{chromatic_mask.size} "
      f"pixels used)")
print(f"(c) mean hue shift, luminance-only equalisation: {d_luma.mean():.2f} deg")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(colour_src)
axes[0].set_title("Original")
axes[1].imshow(per_channel)
axes[1].set_title(f"Per-channel RGB equalised\nmean hue shift={d_pc.mean():.2f}$^\\circ$")
axes[2].imshow(luma_only)
axes[2].set_title(f"Luminance-only equalised\nmean hue shift={d_luma.mean():.2f}$^\\circ$")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(IMG_OUT / "p4_4_2_colour.png", dpi=150)
plt.close(fig)

# ============================================================================
# W1 table
# ============================================================================
tex_path = ROOT / "Report" / "tables_4_2_w1.tex"
with open(tex_path, "w") as f:
    f.write("% Auto-generated by run_4_2_matching.py -- do not edit by hand\n")
    f.write("\\begin{tabular}{lcc}\n\\toprule\n")
    f.write("Target & W1 before & W1 after \\\\\n\\midrule\n")
    for name, before, after in w1_rows:
        f.write(f"{name} & {before:.3f} & {after:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print(f"\nSaved matching/specification/colour figures and W1 table to {IMG_OUT.parent}")
