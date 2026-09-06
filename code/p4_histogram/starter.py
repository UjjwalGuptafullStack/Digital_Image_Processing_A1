"""P4 starter. Implement every function marked TODO.

Dataset map (relative to the bundle root):
  4.1  images/p4/equalization/input.png
  4.2  images/p4/matching/{source,reference}.png
       images/p4/specification/source.png
       images/p4/colour_source.png
  4.3  images/p4/local_regions.png
  4.4  images/p4/exposure/ev0.png ... ev4.png
"""
import numpy as np


def hist(img, bins=256):
    return np.bincount(img.ravel(), minlength=bins).astype(np.float64)


def cdf(h):
    """TODO 4.1: normalised cumulative histogram.

    F[v] = (sum_{u<=v} h[u]) / (sum of all h) -- the INCLUSIVE cumulative
    distribution (h[v] itself counts towards F[v]).
    """
    c = np.cumsum(h)
    total = c[-1]
    return c / total if total > 0 else c


def equalise(img):
    """TODO 4.1: CDF-based global equalisation.

    T(v) = round(255 * F(v)), F the normalised inclusive CDF, no CDF_min
    rescaling (per the handout, for consistent marking).
    """
    img = img.astype(np.uint8)
    h = hist(img, 256)
    F = cdf(h)
    lut = np.clip(np.round(255 * F), 0, 255).astype(np.uint8)
    return lut[img]


def specify(img, target_hist):
    """TODO 4.2: match img's histogram to target_hist.

    Inverse-CDF composition: for each source level v, find the smallest
    output level w such that F_target(w) >= F_source(v) -- the standard
    generalised-inverse construction for matching one discrete CDF to
    another. target_hist need not come from an actual image (a prescribed
    256-bin distribution, e.g. uniform or Gaussian, works the same way).
    """
    img = img.astype(np.uint8)
    h_src = hist(img, 256)
    F_src = cdf(h_src)
    F_tgt = cdf(np.asarray(target_hist, dtype=np.float64))
    # for each source CDF value, first target level whose CDF is >= it
    lut = np.searchsorted(F_tgt, F_src, side="left").clip(0, 255).astype(np.uint8)
    return lut[img]


def wasserstein1(h1, h2):
    """TODO 4.2: W1 distance between two histograms. Your scoring metric.

    For 1-D distributions, W1 has the closed form sum_v |F1(v) - F2(v)|
    (area between the two CDFs); h1, h2 are normalised (or not -- cdf()
    normalises internally) 256-bin histograms over the same support {0..255}.
    """
    F1 = cdf(np.asarray(h1, dtype=np.float64))
    F2 = cdf(np.asarray(h2, dtype=np.float64))
    return float(np.sum(np.abs(F1 - F2)))


def _tile_bounds(size, n_tiles):
    """n_tiles+1 boundary indices splitting [0, size) into n_tiles pieces
    of nearly-equal width (as equal as an integer split allows)."""
    return np.linspace(0, size, n_tiles + 1).round().astype(int)


def ahe(img, tiles=8):
    """TODO 4.3: plain tiled AHE. No clipping, no interpolation. This one is
    SUPPOSED to look bad -- that is the point.

    Independent CDF-based equalisation per tile (same formula as
    `equalise`), each tile's own LUT applied only to its own pixels: no
    smoothing across tile boundaries.
    """
    img = img.astype(np.uint8)
    H, W = img.shape
    ys = _tile_bounds(H, tiles)
    xs = _tile_bounds(W, tiles)
    out = np.zeros_like(img)
    for i in range(tiles):
        for j in range(tiles):
            y0, y1 = ys[i], ys[i + 1]
            x0, x1 = xs[j], xs[j + 1]
            tile = img[y0:y1, x0:x1]
            out[y0:y1, x0:x1] = equalise(tile)
    return out


def _clipped_lut(tile, clip, bins=256):
    """Clip-limited, redistributed CDF-based LUT for one tile.

    Clip each histogram bin at clip*mean_bin_height (mean over `bins` bins,
    including empty ones, so the limit scales with tile pixel count); the
    total excess removed is redistributed uniformly over all bins (a
    second small overflow from that redistribution, if any bin is pushed
    back over the limit, is ignored -- the standard single-pass CLAHE
    approximation). clip=inf disables clipping (reduces to plain AHE on
    that tile).
    """
    h = hist(tile, bins).astype(np.float64)
    n_pixels = tile.size
    mean_height = n_pixels / bins
    limit = clip * mean_height
    if np.isfinite(limit):
        excess = np.sum(np.maximum(h - limit, 0))
        h = np.minimum(h, limit)
        h += excess / bins
    F = cdf(h)
    return np.clip(np.round(255 * F), 0, 255).astype(np.uint8)


def clahe(img, tiles=8, clip=3.0, bins=256):
    """TODO 4.3: clip at `clip` x mean bin height, redistribute the excess,
    and bilinearly interpolate between the four surrounding tile LUTs.

    Watch the tile-centre offset. That is where the marks go.

    Each tile gets its own clip-limited LUT (see _clipped_lut). Every pixel
    is then mapped by BILINEARLY INTERPOLATING between the (up to) four
    nearest TILE-CENTRE LUTs -- not tile corners/edges -- evaluated at that
    pixel's own intensity, with weights given by the pixel's fractional
    position between the surrounding tile centres. Pixels closer to the
    image border than half a tile width, outside the outermost centres,
    are clamped to the nearest centre on that axis (standard CLAHE border
    handling) rather than extrapolated.
    """
    img = img.astype(np.uint8)
    H, W = img.shape
    ys = _tile_bounds(H, tiles)
    xs = _tile_bounds(W, tiles)
    centers_y = (ys[:-1] + ys[1:]) / 2.0
    centers_x = (xs[:-1] + xs[1:]) / 2.0

    luts = np.zeros((tiles, tiles, 256), dtype=np.float64)
    for i in range(tiles):
        for j in range(tiles):
            tile = img[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            luts[i, j] = _clipped_lut(tile, clip, bins)

    # For each pixel, find the tile-centre grid cell it falls in and its
    # fractional position within that cell, clamped to [0, tiles-1] so
    # border pixels (outside the outermost centres) use the nearest tile.
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    fi = np.interp(yy.ravel(), centers_y, np.arange(tiles)).reshape(H, W)
    fj = np.interp(xx.ravel(), centers_x, np.arange(tiles)).reshape(H, W)
    fi = np.clip(fi, 0, tiles - 1)
    fj = np.clip(fj, 0, tiles - 1)

    i0 = np.floor(fi).astype(int).clip(0, tiles - 1)
    i1 = np.clip(i0 + 1, 0, tiles - 1)
    j0 = np.floor(fj).astype(int).clip(0, tiles - 1)
    j1 = np.clip(j0 + 1, 0, tiles - 1)
    ti = (fi - i0)
    tj = (fj - j0)

    v = img  # pixel intensities, used to index into each surrounding LUT

    def gather(lut_i, lut_j):
        # luts[lut_i, lut_j, v]: lut_i, lut_j, v all broadcast over (H, W)
        return luts[lut_i, lut_j, v]

    L00 = gather(i0, j0)
    L01 = gather(i0, j1)
    L10 = gather(i1, j0)
    L11 = gather(i1, j1)

    top = L00 * (1 - tj) + L01 * tj
    bot = L10 * (1 - tj) + L11 * tj
    out = top * (1 - ti) + bot * ti
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def apply_on_luma(rgb, fn):
    """TODO 4.2: run a greyscale operator on luma only, preserving hue.

    Converts to BT.601 luma Y, runs fn (e.g. equalise) on Y alone, then
    rescales R,G,B by the ratio new_Y/old_Y at each pixel -- this changes
    brightness while leaving each pixel's relative R:G:B ratios (hence
    hue and saturation) unchanged, unlike equalising R,G,B independently.
    """
    rgb = rgb.astype(np.float64)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Y_u8 = np.clip(np.round(Y), 0, 255).astype(np.uint8)
    Y_new = fn(Y_u8).astype(np.float64)

    ratio = np.where(Y > 1e-6, Y_new / np.maximum(Y, 1e-6), 1.0)
    out = rgb * ratio[..., None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def auto_correct(rgb):
    """TODO 4.4: no parameters. Estimate what you need from the image.

    Estimator: percentile-based linear contrast stretch on BT.601 luma.
    p1, p99 = the image's own 1st and 99th luma percentiles (estimated
    fresh from each image, not a fixed constant) are mapped to 0 and 255
    respectively; applied to R,G,B via the same luma-ratio rescaling as
    apply_on_luma, so hue is preserved and only brightness/contrast
    changes. This is parameter-free in the sense the handout means
    (nothing is tuned per image by a human; p1/p99 ARE the per-image
    estimates the function is required to compute internally).

    Failure case (see report): if an image's luma is already saturated
    over a large fraction of pixels (e.g. badly overexposed, most pixels
    at 255), p1 and p99 can coincide or nearly coincide, making the
    stretch degenerate (a near-infinite or clamped gain). This is guarded
    against below by falling back to no-op (gain 1) when p99-p1 is too
    small to trust, rather than dividing by near-zero and amplifying
    noise -- but the guard's very presence is the honest admission that
    such inputs cannot be corrected by this class of estimator; the
    information needed (what the true bright detail should look like) is
    not recoverable from a clipped image.
    """
    rgb = rgb.astype(np.float64)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B

    p1, p99 = np.percentile(Y, [1, 99])
    spread = p99 - p1
    if spread < 10.0:  # degenerate: cannot estimate a reliable stretch
        return np.clip(np.round(rgb), 0, 255).astype(np.uint8)

    Y_new = np.clip((Y - p1) * (255.0 / spread), 0, 255)
    ratio = np.where(Y > 1e-6, Y_new / np.maximum(Y, 1e-6), 1.0)
    out = rgb * ratio[..., None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
