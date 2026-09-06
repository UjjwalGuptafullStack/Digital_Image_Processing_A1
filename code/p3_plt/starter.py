"""P3 starter. Implement every function marked TODO.

The transform is pointwise. That is the whole hint.
"""
import numpy as np


def transfer_curve(img_in, img_out):
    """TODO 3.1: estimate T[v] for v in 0..255, plus the pixel count supporting
    each level. Levels with no pixels must be distinguishable from levels
    that map to zero -- you need both arrays.

    For each input level v, T[v] is estimated as the mode of the output
    values at pixels where img_in == v (the mode rather than the mean
    because the transform is a deterministic LUT -- every pixel at level v
    maps to the identical output level whenever v is sampled at all -- so
    the mode recovers that exact value and is robust to the rare case of
    any noise/antialiasing pixels). cnt[v] is the number of supporting
    pixels; cnt[v] == 0 marks a level the input never samples, so T[v] for
    that v (returned as 0 as a value with no support) must not be trusted
    -- callers must check cnt, not just read T.
    """
    img_in = img_in.astype(np.uint8)
    img_out = img_out.astype(np.uint8)
    T = np.zeros(256, dtype=np.float64)
    cnt = np.zeros(256, dtype=np.int64)
    flat_in = img_in.flatten()
    flat_out = img_out.flatten()
    order = np.argsort(flat_in, kind="stable")
    sorted_in = flat_in[order]
    sorted_out = flat_out[order]
    boundaries = np.searchsorted(sorted_in, np.arange(257))
    for v in range(256):
        lo, hi = boundaries[v], boundaries[v + 1]
        if hi > lo:
            vals = sorted_out[lo:hi]
            counts = np.bincount(vals, minlength=256)
            T[v] = np.argmax(counts)
            cnt[v] = hi - lo
    return T, cnt


def _seg_sse_and_fit(vs, ys, lo, hi):
    """Unweighted least-squares line through points [lo, hi) of (vs, ys).
    Unweighted (one vote per supported LEVEL, not per pixel): repeated
    pixels at the same input level are the same fact about T observed many
    times, not independent evidence, so weighting by pixel count would let
    a few heavily-populated levels dominate the fit and would also blow up
    the effective sample size used for the BIC penalty below."""
    x = vs[lo:hi]
    y = ys[lo:hi]
    if hi - lo < 1:
        return 0.0, 0.0, 0.0
    if hi - lo == 1 or np.all(x == x[0]):
        return 0.0, 0.0, float(y[0])
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    slope = ((x - xm) * (y - ym)).sum() / sxx if sxx > 1e-9 else 0.0
    intercept = ym - slope * xm
    pred = slope * x + intercept
    sse = float(((y - pred) ** 2).sum())
    return sse, float(slope), float(intercept)


def fit_piecewise(T, cnt, max_seg=6):
    """TODO 3.2: fit the smallest number of segments the data justifies.

    Only levels with cnt[v] > 0 constrain the fit (unsupported levels carry
    no information about T, see support_report). For each candidate number
    of segments k = 1..max_seg, an exact dynamic program finds the
    breakpoint placement (over supported levels only) minimising total
    unweighted SSE -- this handles non-monotonic transforms (e.g. B) the
    same as monotonic ones, since each segment is just an independent
    least-squares line.

    Complexity penalty: BIC, n*log(SSE_floored/n) + n_params*log(n), with
    n = number of supported levels (not pixel count -- see
    _seg_sse_and_fit) and n_params = 2 per segment (slope, intercept).
    SSE is floored at n/12 before taking the log: 1/12 is the variance of
    +-0.5 uniform rounding noise in an integer-valued output LUT, i.e. the
    best SSE any model can meaningfully achieve once the residual is pure
    quantisation. Without this floor, BIC's log(SSE) term keeps rewarding
    extra segments that only chase quantisation noise (see the discussion
    in the report of transform A, where segment counts above 3 split one
    true segment into two near-identical slope-4 pieces); with the floor,
    once SSE is already at or below the noise floor, only the k-dependent
    parameter penalty n_params*log(n) can still change, so BIC correctly
    stops preferring more segments.

    Returns (breakpoints, slopes, intercepts, n_segments), where
    breakpoints has n_segments+1 entries (supported-level endpoints).
    """
    supported = cnt > 0
    vs = np.where(supported)[0].astype(np.float64)
    ys = T[supported]
    n = len(vs)
    if n < 2:
        raise ValueError("fit_piecewise: fewer than 2 supported levels")
    max_seg = min(max_seg, n)

    # cost[lo][hi] = SSE of the best line through vs[lo:hi], ys[lo:hi]
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
    best_bic, best_k, best_segs = np.inf, None, None
    for k in range(1, max_seg + 1):
        sse, segs = dp_fit(k)
        n_params = 2 * k
        sse_floored = max(sse, n * quant_floor_mse)
        bic = n * np.log(sse_floored / n) + n_params * np.log(n)
        if bic < best_bic:
            best_bic, best_k, best_segs = bic, k, segs

    breakpoints = [int(vs[lo]) for lo, _ in best_segs] + [int(vs[best_segs[-1][1] - 1])]
    slopes, intercepts = [], []
    for lo, hi in best_segs:
        _, slope, intercept = _seg_sse_and_fit(vs, ys, lo, hi)
        slopes.append(slope)
        intercepts.append(intercept)
    return np.array(breakpoints), np.array(slopes), np.array(intercepts), best_k


def apply_recovered(img, bps, slopes, inter):
    """TODO 3.1: build the LUT from your fit and apply it.

    bps: n_segments+1 breakpoints (bps[0]=0, bps[-1]=255, increasing).
    slopes[i], inter[i]: segment i (levels in [bps[i], bps[i+1]]) is
    T(v) = slopes[i]*v + inter[i]. Builds a length-256 lookup table by
    evaluating the appropriate segment for each level, then applies it by
    indexing (a LUT application, not a per-pixel Python loop).
    """
    lut = np.zeros(256, dtype=np.float64)
    n_seg = len(slopes)
    for v in range(256):
        seg = np.searchsorted(bps, v, side="right") - 1
        seg = min(max(seg, 0), n_seg - 1)
        lut[v] = slopes[seg] * v + inter[seg]
    lut = np.clip(np.round(lut), 0, 255).astype(np.uint8)
    return lut[img.astype(np.uint8)]


def support_report(cnt, min_count=10, **kw):
    """TODO 3.3: which intensity ranges does the input not sample well enough
    to constrain the transform? Return something human-readable.

    A level v is "trusted" if cnt[v] >= min_count. Below that, T[v] is
    either fully unidentifiable (cnt[v] == 0: no pixel at that level, so
    the data says nothing about T[v] whatsoever) or estimated from so few
    pixels that the estimate is not reliable evidence for a segment
    boundary. Returns a dict summarising, as contiguous ranges, which
    levels are unsupported (cnt == 0), which are trusted (cnt >=
    min_count), and which are in between (weakly supported).
    """
    def ranges(mask):
        out = []
        v = 0
        while v < len(mask):
            if mask[v]:
                start = v
                while v < len(mask) and mask[v]:
                    v += 1
                out.append((start, v - 1))
            else:
                v += 1
        return out

    unsupported = cnt == 0
    weak = (cnt > 0) & (cnt < min_count)
    trusted = cnt >= min_count

    return {
        "min_count": min_count,
        "unsupported_ranges": ranges(unsupported),
        "weak_ranges": ranges(weak),
        "trusted_ranges": ranges(trusted),
        "n_unsupported": int(unsupported.sum()),
        "n_weak": int(weak.sum()),
        "n_trusted": int(trusted.sum()),
    }
