"""1.4 Runtime analysis: two log-log sweeps with fitted slopes (runtime vs k
at N=512, runtime vs N at k=15), im2col peak-memory table, and the separable
speedup measurement at 512^2 and 2048^2 needed for the written discussion.
All timings use the separable Gaussian kernel (rank 1 at every k), matching
the kernel used for the 1.1 correctness table.
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from starter import (kernel_bank, conv2d_taps, conv2d_im2col, conv2d_fft,
                      conv2d_separable)

ROOT = Path(__file__).resolve().parents[2]
IMG_OUT = ROOT / "Report" / "Images"
IMG_OUT.mkdir(parents=True, exist_ok=True)

full = np.asarray(Image.open(ROOT / "images/p1/base_2048.png")).astype(float)


def time_fn(fn, img, K, reps=7):
    """Median of `reps` timings. Median (not best-of-N) is used because
    best-of-N timings at small k/N are dominated by occasional very fast
    outlier runs (OS/GC noise), which distorts the fitted log-log slope;
    the median is the standard robust choice for wall-clock microbenchmarks."""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(img, K)
        t1 = time.perf_counter()
        ts.append(t1 - t0)
    ts.sort()
    return ts[len(ts) // 2]


def fit_loglog_slope(xs, ys):
    """Least-squares slope of log(y) vs log(x)."""
    lx, ly = np.log(xs), np.log(ys)
    slope, intercept = np.polyfit(lx, ly, 1)
    return slope, intercept


# ============================================================================
# Sweep 1: runtime vs k, N=512 fixed
# ============================================================================
N_FIXED = 512
K_VALUES = [3, 7, 11, 15, 21, 31]
crop512 = full[:N_FIXED, :N_FIXED]

print(f"=== Sweep 1: runtime vs k, N={N_FIXED} ===")
times_taps_k, times_im2col_k, times_fft_k = [], [], []
for k in K_VALUES:
    K = kernel_bank(k)["gaussian"]
    t_taps = time_fn(conv2d_taps, crop512, K)
    t_im2col = time_fn(conv2d_im2col, crop512, K)
    t_fft = time_fn(conv2d_fft, crop512, K)
    times_taps_k.append(t_taps)
    times_im2col_k.append(t_im2col)
    times_fft_k.append(t_fft)
    print(f"k={k:3d}  taps={t_taps:8.4f}s  im2col={t_im2col:8.4f}s  fft={t_fft:8.4f}s")

slope_taps_k, _ = fit_loglog_slope(K_VALUES, times_taps_k)
slope_im2col_k, _ = fit_loglog_slope(K_VALUES, times_im2col_k)
print(f"fitted slope (taps) vs k: {slope_taps_k:.3f}  (theory: 2.0)")
print(f"fitted slope (im2col) vs k: {slope_im2col_k:.3f}")

# ============================================================================
# Sweep 2: runtime vs N, k=15 fixed
# ============================================================================
K_FIXED = 15
N_VALUES = [128, 256, 512, 1024, 2048]
Kgauss15 = kernel_bank(K_FIXED)["gaussian"]

print(f"\n=== Sweep 2: runtime vs N, k={K_FIXED} ===")
times_taps_N, times_im2col_N, times_fft_N = [], [], []
peak_mem_N = []
for N in N_VALUES:
    # Fewer reps for large N: absolute times are already large (seconds),
    # so relative timing noise matters less and 7 reps would be very slow.
    reps = 7 if N <= 512 else 3
    crop = full[:N, :N]
    t_taps = time_fn(conv2d_taps, crop, Kgauss15, reps=reps)
    t_im2col = time_fn(conv2d_im2col, crop, Kgauss15, reps=reps)
    t_fft = time_fn(conv2d_fft, crop, Kgauss15, reps=reps)
    times_taps_N.append(t_taps)
    times_im2col_N.append(t_im2col)
    times_fft_N.append(t_fft)
    # peak im2col patch-matrix memory: (N*N, k*k) float64 array
    peak_bytes = N * N * K_FIXED * K_FIXED * 8
    peak_mem_N.append(peak_bytes)
    print(f"N={N:5d}  taps={t_taps:9.4f}s  im2col={t_im2col:9.4f}s  fft={t_fft:9.4f}s  "
          f"im2col_mem={peak_bytes/1e6:9.1f}MB")

slope_taps_N, _ = fit_loglog_slope(N_VALUES, times_taps_N)
slope_im2col_N, _ = fit_loglog_slope(N_VALUES, times_im2col_N)
print(f"fitted slope (taps) vs N: {slope_taps_N:.3f}  (theory: 2.0, since H*W ~ N^2)")
print(f"fitted slope (im2col) vs N: {slope_im2col_N:.3f}")

# ============================================================================
# Part (c): separable speedup at 512^2 and 2048^2, k=15
# ============================================================================
print(f"\n=== Separable speedup, k={K_FIXED} ===")
speedups = {}
for N in (512, 2048):
    crop = full[:N, :N]
    t_taps = time_fn(conv2d_taps, crop, Kgauss15, reps=3)
    t_sep = time_fn(conv2d_separable, crop, Kgauss15, reps=3)
    speedup = t_taps / t_sep
    speedups[N] = (t_taps, t_sep, speedup)
    print(f"N={N:5d}  taps={t_taps:8.4f}s  separable={t_sep:8.4f}s  "
          f"speedup={speedup:.2f}x  (theory: k/2={K_FIXED/2:.1f}x)")

# ============================================================================
# Plots
# ============================================================================
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(K_VALUES, times_taps_k, "o-", label=f"conv2d_taps (slope={slope_taps_k:.2f})")
ax.loglog(K_VALUES, times_im2col_k, "s-", label=f"conv2d_im2col (slope={slope_im2col_k:.2f})")
ax.loglog(K_VALUES, times_fft_k, "^-", label="conv2d_fft")
ax.set_xlabel("$k$ (kernel size)")
ax.set_ylabel("runtime (s)")
ax.set_title(f"Runtime vs. $k$, $N={N_FIXED}$ (log--log)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(IMG_OUT / "p1_1_4_runtime_vs_k.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(N_VALUES, times_taps_N, "o-", label=f"conv2d_taps (slope={slope_taps_N:.2f})")
ax.loglog(N_VALUES, times_im2col_N, "s-", label=f"conv2d_im2col (slope={slope_im2col_N:.2f})")
ax.loglog(N_VALUES, times_fft_N, "^-", label="conv2d_fft")
ax.set_xlabel("$N$ (image side length)")
ax.set_ylabel("runtime (s)")
ax.set_title(f"Runtime vs. $N$, $k={K_FIXED}$ (log--log)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(IMG_OUT / "p1_1_4_runtime_vs_N.png", dpi=150)
plt.close(fig)

# ============================================================================
# Tables
# ============================================================================
tex_path = ROOT / "Report" / "tables_1_4_memory.tex"
with open(tex_path, "w") as f:
    f.write("% Auto-generated by run_1_4_runtime.py -- do not edit by hand\n")
    f.write("\\begin{tabular}{lccc}\n\\toprule\n")
    f.write("$N$ & patch matrix shape & peak memory (float64) & as fraction of image size \\\\\n\\midrule\n")
    for N, mem in zip(N_VALUES, peak_mem_N):
        img_bytes = N * N * 8
        f.write(f"{N} & $({N*N},\\,{K_FIXED*K_FIXED})$ & {mem/1e6:.1f} MB & "
                f"{mem/img_bytes:.0f}$\\times$ \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print(f"\nSaved memory table to {tex_path}")

speed_tex_path = ROOT / "Report" / "tables_1_4_speedup.tex"
with open(speed_tex_path, "w") as f:
    f.write("% Auto-generated by run_1_4_runtime.py -- do not edit by hand\n")
    f.write("\\begin{tabular}{lccc}\n\\toprule\n")
    f.write("$N$ & \\texttt{conv2d\\_taps} & \\texttt{conv2d\\_separable} & measured speedup ($k/2=7.5\\times$ theory) \\\\\n\\midrule\n")
    for N in (512, 2048):
        t_taps, t_sep, speedup = speedups[N]
        f.write(f"{N} & {t_taps:.4f}s & {t_sep:.4f}s & {speedup:.2f}$\\times$ \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print(f"Saved speedup table to {speed_tex_path}")

print(f"\nSaved plots to {IMG_OUT}")
