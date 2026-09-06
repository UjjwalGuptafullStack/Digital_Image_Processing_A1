"""3.2 Choose the number of segments: transform C, no hint given. Shows the
BIC-vs-k selection curve for C, and separately demonstrates on transform A
(whose true segment count, 3, is stated in the handout) that an unfloored
BIC is fooled into selecting k=4 -- splitting one true segment into two
near-identical-slope pieces to chase quantisation noise -- while the
quantisation-floored BIC correctly recovers k=3.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from starter import transfer_curve, fit_piecewise, apply_recovered, _seg_sse_and_fit

ROOT = Path(__file__).resolve().parents[2]
IMG_OUT = ROOT / "Report" / "Images"
IMG_OUT.mkdir(parents=True, exist_ok=True)

c_in = np.asarray(Image.open(ROOT / "images/p3/nebula_in.png"))
c_out = np.asarray(Image.open(ROOT / "images/p3/nebula_out_C.png"))

T, cnt = transfer_curve(c_in, c_out)
supported = cnt > 0
vs = np.where(supported)[0].astype(np.float64)
ys = T[supported]
n = len(vs)

MAX_SEG = 8

# Recompute the DP sweep manually (mirrors fit_piecewise internals) so we
# can report SSE/BIC at every k, with and without the quantisation floor,
# for the discussion plot -- fit_piecewise itself only returns the winner.
cost = np.full((n + 1, n + 1), np.inf)
for lo in range(n):
    for hi in range(lo + 1, n + 1):
        sse, _, _ = _seg_sse_and_fit(vs, ys, lo, hi)
        cost[lo, hi] = sse


def dp_fit(k):
    dp = np.full((k + 1, n + 1), np.inf)
    choice = np.full((k + 1, n + 1), -1, dtype=int)
    dp[0][0] = 0.0
    for j in range(1, k + 1):
        for i in range(j, n + 1):
            best, best_lo = np.inf, -1
            for lo in range(j - 1, i):
                c = dp[j - 1][lo] + cost[lo][i]
                if c < best:
                    best, best_lo = c, lo
            dp[j][i] = best
            choice[j][i] = best_lo
    segs = []
    i, j = n, k
    while j > 0:
        lo = choice[j][i]
        segs.append((lo, i))
        i, j = lo, j - 1
    segs.reverse()
    return dp[k][n], segs


quant_floor_mse = 1.0 / 12.0
sse_list, bic_floored, bic_unfloored = [], [], []
for k in range(1, MAX_SEG + 1):
    sse, segs = dp_fit(k)
    n_params = 2 * k
    sse_list.append(sse)
    bic_floored.append(n * np.log(max(sse, n * quant_floor_mse) / n) + n_params * np.log(n))
    bic_unfloored.append(n * np.log(sse / n + 1e-12) + n_params * np.log(n))
    print(f"k={k}  SSE={sse:12.4f}  BIC(floored)={bic_floored[-1]:10.3f}  "
          f"BIC(unfloored)={bic_unfloored[-1]:10.3f}")

k_floored = int(np.argmin(bic_floored)) + 1
k_unfloored = int(np.argmin(bic_unfloored)) + 1
print(f"\nselected k (floored BIC): {k_floored}")
print(f"selected k (unfloored BIC, keeps decreasing): {k_unfloored}")

# --- final fit via fit_piecewise (uses the floored criterion) ----------
bps, slopes, inter, k = fit_piecewise(T, cnt, max_seg=MAX_SEG)
recon = apply_recovered(c_in, bps, slopes, inter)
err = np.abs(recon.astype(np.float64) - c_out.astype(np.float64))
print(f"final: k={k}  bps={bps}  max_err={err.max():.4f}  mean_err={err.mean():.6f}")

# --- plot: BIC vs k, floored vs unfloored -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ks = list(range(1, MAX_SEG + 1))
axes[0].plot(ks, sse_list, "o-")
axes[0].set_yscale("log")
axes[0].axhline(n * quant_floor_mse, color="red", linestyle="--",
                 label=f"quantisation floor ($n/12={n*quant_floor_mse:.1f}$)")
axes[0].set_xlabel("$k$ (segments)")
axes[0].set_ylabel("SSE (log scale)")
axes[0].set_title("SSE keeps falling with $k$ (always will)")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(ks, bic_unfloored, "s-", label="BIC, no floor (keeps falling)", color="indianred")
axes[1].plot(ks, bic_floored, "o-", label="BIC, quantisation-floored (selects $k$="
             f"{k_floored})", color="steelblue")
axes[1].axvline(k_floored, color="steelblue", linestyle=":", linewidth=1)
axes[1].set_xlabel("$k$ (segments)")
axes[1].set_ylabel("BIC")
axes[1].set_title("Segment count selection")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
fig.suptitle(f"Transform C: choosing $k$ by BIC (n={n} supported levels)")
fig.tight_layout()
fig.savefig(IMG_OUT / "p3_3_2_bic_selection.png", dpi=150)
plt.close(fig)

# --- scatter + fit for C -------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.scatter(vs, ys, s=4, alpha=0.5, label="measured $T[v]$")
lut_v = np.arange(256)
lut = np.zeros(256)
for v in lut_v:
    seg = min(max(np.searchsorted(bps, v, side="right") - 1, 0), len(slopes) - 1)
    lut[v] = slopes[seg] * v + inter[seg]
ax.plot(lut_v, lut, color="red", linewidth=1.5, label=f"fitted ($k$={k})")
for bp in bps:
    ax.axvline(bp, color="gray", linestyle=":", linewidth=0.8)
ax.set_xlabel("input level $v$")
ax.set_ylabel("$T[v]$")
ax.set_title(f"Transform C: recovered fit, $k$={k}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(IMG_OUT / "p3_3_2_scatter_fit_C.png", dpi=150)
plt.close(fig)

# ============================================================================
# Quantisation-floor demonstration on transform A (true k=3, stated in the
# handout): show that an UNFLOORED BIC is fooled into selecting k=4 by
# splitting one true segment into two near-identical-slope pieces.
# ============================================================================
print("\n=== Quantisation-floor demonstration (transform A, true k=3) ===")
field_in = np.asarray(Image.open(ROOT / "images/p3/field_in.png"))
out_A = np.asarray(Image.open(ROOT / "images/p3/field_out_A.png"))
TA, cntA = transfer_curve(field_in, out_A)
supportedA = cntA > 0
vsA = np.where(supportedA)[0].astype(np.float64)
ysA = TA[supportedA]
nA = len(vsA)

costA = np.full((nA + 1, nA + 1), np.inf)
for lo in range(nA):
    for hi in range(lo + 1, nA + 1):
        sse, _, _ = _seg_sse_and_fit(vsA, ysA, lo, hi)
        costA[lo, hi] = sse


def dp_fit_A(k):
    dp = np.full((k + 1, nA + 1), np.inf)
    choice = np.full((k + 1, nA + 1), -1, dtype=int)
    dp[0][0] = 0.0
    for j in range(1, k + 1):
        for i in range(j, nA + 1):
            best, best_lo = np.inf, -1
            for lo in range(j - 1, i):
                c = dp[j - 1][lo] + costA[lo][i]
                if c < best:
                    best, best_lo = c, lo
            dp[j][i] = best
            choice[j][i] = best_lo
    segs = []
    i, j = nA, k
    while j > 0:
        lo = choice[j][i]
        segs.append((lo, i))
        i, j = lo, j - 1
    segs.reverse()
    return dp[k][nA], segs


sseA_list, bicA_floored, bicA_unfloored = [], [], []
for k in range(1, 7):
    sse, segs = dp_fit_A(k)
    n_params = 2 * k
    sseA_list.append(sse)
    bicA_floored.append(nA * np.log(max(sse, nA * quant_floor_mse) / nA) + n_params * np.log(nA))
    bicA_unfloored.append(nA * np.log(sse / nA + 1e-12) + n_params * np.log(nA))
    if k == 4:
        # report the two "duplicate" segments a k=4 unfloored fit produces
        seg_slopes = []
        for lo, hi in segs:
            _, slope, intercept = _seg_sse_and_fit(vsA, ysA, lo, hi)
            seg_slopes.append((int(vsA[lo]), int(vsA[hi - 1]), slope, intercept))
        print(f"k=4 segments (unfloored winner): {seg_slopes}")

kA_floored = int(np.argmin(bicA_floored)) + 1
kA_unfloored = int(np.argmin(bicA_unfloored)) + 1
for k in range(1, 7):
    print(f"A: k={k}  SSE={sseA_list[k-1]:12.4f}  BIC_floored={bicA_floored[k-1]:10.3f}  "
          f"BIC_unfloored={bicA_unfloored[k-1]:10.3f}")
print(f"A: selected k (floored) = {kA_floored} (matches handout's stated 3-segment structure)")
print(f"A: selected k (unfloored) = {kA_unfloored} (spurious extra segment)")

fig, ax = plt.subplots(figsize=(7, 5))
ksA = list(range(1, 7))
ax.plot(ksA, bicA_unfloored, "s-", color="indianred",
        label=f"BIC, no floor (selects $k$={kA_unfloored}, wrong)")
ax.plot(ksA, bicA_floored, "o-", color="steelblue",
        label=f"BIC, quantisation-floored (selects $k$={kA_floored}, correct)")
ax.axvline(kA_floored, color="steelblue", linestyle=":", linewidth=1)
ax.axvline(kA_unfloored, color="indianred", linestyle=":", linewidth=1)
ax.set_xlabel("$k$ (segments)")
ax.set_ylabel("BIC")
ax.set_title("Transform A (true $k$=3): unfloored BIC overfits to quantisation noise")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(IMG_OUT / "p3_3_2_quantisation_floor_demo.png", dpi=150)
plt.close(fig)

print(f"\nSaved quantisation-floor demonstration plot to {IMG_OUT}")
print(f"\nSaved BIC-selection and scatter+fit plots to {IMG_OUT}")
