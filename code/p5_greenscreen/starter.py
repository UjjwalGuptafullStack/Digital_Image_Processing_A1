"""P5 starter. Implement every function marked TODO."""
import numpy as np


def to_ycbcr(rgb):
    """TODO 5.1: BT.601 conversion.

    Full-range digital BT.601: Y in [0,255], Cb/Cr centred on 128.
    """
    rgb = rgb.astype(np.float64)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128.0 + (B - Y) * 0.564
    Cr = 128.0 + (R - Y) * 0.713
    return np.stack([Y, Cb, Cr], axis=-1)


def key_naive_rgb(rgb, thresh=60):
    """TODO 5.1: green dominance in raw RGB. Expected to fail on the unevenly
    lit backing -- show where.

    Backing (to be keyed OUT, alpha=0) is where green dominates both other
    channels by more than `thresh`: G - max(R,B) > thresh. Returns a hard
    (0/1) alpha map, 1 = foreground.
    """
    rgb = rgb.astype(np.float64)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    is_backing = (G - np.maximum(R, B)) > thresh
    return (~is_backing).astype(np.float64)


def estimate_key_colour(rgb, border=24):
    """TODO 5.2: estimate the backing chroma from the frame. Do not hard-code
    a green value; the graded footage varies.

    Samples a `border`-pixel-wide frame around the image edges (assumed to
    be backing, since a correctly framed subject does not touch the image
    boundary) and returns the MEDIAN (Cb, Cr) over those pixels -- median
    rather than mean so a few subject/prop pixels that happen to touch the
    border do not pull the estimate away from the dominant backing colour.
    """
    H, W, _ = rgb.shape
    top = rgb[:border, :, :]
    bottom = rgb[H - border:, :, :]
    left = rgb[:, :border, :]
    right = rgb[:, W - border:, :]
    border_pixels = np.concatenate([
        top.reshape(-1, 3), bottom.reshape(-1, 3),
        left.reshape(-1, 3), right.reshape(-1, 3),
    ], axis=0)
    ycc = to_ycbcr(border_pixels[None, :, :])[0]
    return np.median(ycc[:, 1:], axis=0)  # (Cb, Cr)


def key_soft(rgb, key_cbcr=None, t_in=None, t_out=None):
    """TODO 5.2: fractional alpha from chroma distance. Choose t_in and t_out
    from your own data -- look at the distance distribution for pure backing
    versus pure foreground before you pick numbers.

    chroma distance d(pixel) = ||(Cb,Cr) - key_cbcr||. Alpha is a double
    threshold ramp: d <= t_in -> alpha=0 (pure backing), d >= t_out ->
    alpha=1 (pure foreground), linear in between (fractional alpha for
    pixels that are a real optical mix of foreground and backing, e.g. a
    hair strand or motion-blurred edge).
    """
    if key_cbcr is None:
        key_cbcr = estimate_key_colour(rgb)
    if t_in is None:
        t_in = 12.0  # just above measured pure-backing chroma-distance p99 (~8.6)
    if t_out is None:
        t_out = 58.0  # at measured pure-foreground chroma-distance p1 (~62, small margin)

    ycc = to_ycbcr(rgb)
    Cb, Cr = ycc[..., 1], ycc[..., 2]
    dist = np.sqrt((Cb - key_cbcr[0]) ** 2 + (Cr - key_cbcr[1]) ** 2)

    alpha = (dist - t_in) / (t_out - t_in)
    return np.clip(alpha, 0.0, 1.0)


def suppress_spill(rgb, alpha, strength=1.0, key_cbcr=None):
    """TODO 5.2: remove green bounce. Should scale with (1 - alpha). Why?

    Spill is green light bouncing off the backing onto the subject's edges
    (e.g. hair, skin near the silhouette). It shows up as excess G over
    max(R,B). Correction: clamp G down to max(R,B) (i.e. remove exactly the
    green excess), then blend that correction in proportional to (1-alpha).

    Scaling by (1-alpha) matters because alpha is itself an estimate of how
    much backing light is mixed into this pixel: a pixel with alpha=1 is
    judged pure, confidently-identified foreground (its "excess green", if
    any, is genuine subject colour, e.g. a green shirt, and must not be
    touched), while a pixel with alpha=0 is pure backing and is about to be
    replaced by the background pixel entirely, so correcting its RGB is
    moot. Spill contamination is concentrated exactly at fractional-alpha
    pixels -- partial mixtures of subject and backing light -- so the
    correction should be strongest there and vanish at both alpha extremes.
    """
    rgb = rgb.astype(np.float64)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    excess = np.maximum(G - np.maximum(R, B), 0.0)
    G_desp = G - strength * excess * (1.0 - alpha)
    return np.clip(np.stack([R, G_desp, B], axis=-1), 0, 255).astype(np.uint8)


def composite(fg, bg, alpha, spill=True, key_cbcr=None, spill_strength=1.0):
    """TODO 5.2: I = a*F + (1-a)*B.

    fg, bg: (H,W,3) uint8, same shape. alpha: (H,W) in [0,1]. If spill,
    de-spills fg (scaled by (1-alpha), see suppress_spill) before blending,
    so spill-contaminated pixels are cleaned before being weighted into the
    output rather than after.
    """
    fg_use = suppress_spill(fg, alpha, strength=spill_strength, key_cbcr=key_cbcr) if spill else fg
    fg_use = fg_use.astype(np.float64)
    bg_use = bg.astype(np.float64)
    a = alpha[..., None]
    out = a * fg_use + (1.0 - a) * bg_use
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


class ChromaLUT:
    """TODO 5.3: quantise (Cb,Cr) so keying is one table lookup per pixel."""

    def __init__(self, key_cbcr, t_in=None, t_out=None, bins=128):
        raise NotImplementedError

    def alpha(self, rgb):
        raise NotImplementedError


# ---- provided metrics: do not modify
def sad(a, b):
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).sum() / 1000.0)


def mse_alpha(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64))**2))


def grad_error(a, b):
    from scipy.ndimage import gaussian_filter
    ga = np.hypot(*np.gradient(gaussian_filter(a.astype(np.float64), 1.0)))
    gb = np.hypot(*np.gradient(gaussian_filter(b.astype(np.float64), 1.0)))
    return float(np.abs(ga - gb).sum() / 1000.0)


def temporal_flicker(alphas):
    if len(alphas) < 2:
        return 0.0
    total = 0.0
    count = 0
    prev = np.asarray(alphas[0], dtype=np.float32)
    for alpha in alphas[1:]:
        cur = np.asarray(alpha, dtype=np.float32)
        total += float(np.abs(cur - prev).sum(dtype=np.float64))
        count += cur.size
        prev = cur
    return total / count
