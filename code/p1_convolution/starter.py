"""P1 starter. Implement every function marked TODO.

Rules: NumPy array arithmetic only. scipy/cv2/skimage may be used to CHECK
your answers, never to produce them. numpy.fft is allowed in conv2d_fft.
"""
import numpy as np


def kernel_bank(k=15):
    """Provided. Do not modify -- your rank table must match these kernels."""
    ax = np.arange(k) - (k - 1) / 2
    box = np.ones((k, k)) / (k * k)
    s = k / 6.0
    g1 = np.exp(-(ax**2) / (2 * s * s)); g1 /= g1.sum()
    gauss = np.outer(g1, g1)
    sobel = np.outer([1, 2, 1], [-1, 0, 1]).astype(float)
    xx, yy = np.meshgrid(ax, ax); r2 = xx**2 + yy**2
    log = (r2 - 2*s*s) / (s**4) * np.exp(-r2 / (2*s*s))
    motion0 = np.zeros((k, k)); motion0[k // 2, :] = 1.0 / k
    disk = (r2 <= (k/2.0)**2).astype(float); disk /= disk.sum()
    rand = np.random.default_rng(0).normal(size=(k, k)); rand /= np.abs(rand).sum()
    return {"box": box, "gaussian": gauss, "sobel3": sobel, "log": log,
            "log_dc_removed": log - log.mean(), "motion_0deg": motion0,
            "motion_45deg": np.eye(k) / k, "disk": disk, "random": rand}


def numeric_rank(K, tol=None):
    """Numerical rank from the singular values.

    A singular value sigma_i is treated as zero if sigma_i <= tol, where by
    default tol = max(K.shape) * sigma_max * eps(float64). This is the same
    scale-relative convention numpy.linalg.matrix_rank uses: it is relative
    to sigma_max (so uniformly scaling K does not change the reported rank)
    and grows with matrix size (because SVD round-off accumulates roughly
    with dimension). A fixed absolute tolerance would fail both properties.
    """
    s = np.linalg.svd(K, compute_uv=False)
    if tol is None:
        eps = np.finfo(K.dtype if np.issubdtype(K.dtype, np.floating) else np.float64).eps
        tol = max(K.shape) * s.max() * eps
    return int(np.sum(s > tol)), s, tol


def conv2d_loops(img, K):
    """TODO 1.1: four nested Python loops. Zero-padded, 'same', TRUE convolution.
    Only run this on small crops -- see the handout."""
    H, W = img.shape
    kh, kw = K.shape
    ph, pw = kh // 2, kw // 2
    padded = np.zeros((H + 2 * ph, W + 2 * pw), dtype=np.float64)
    padded[ph:ph + H, pw:pw + W] = img

    Kflip = K[::-1, ::-1]

    out = np.zeros((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            acc = 0.0
            for m in range(kh):
                for n in range(kw):
                    acc += padded[i + m, j + n] * Kflip[m, n]
            out[i, j] = acc
    return out


def conv2d_taps(img, K):
    """TODO 1.1: loop over kernel taps, vectorised over the image."""
    H, W = img.shape
    kh, kw = K.shape
    ph, pw = kh // 2, kw // 2
    padded = np.zeros((H + 2 * ph, W + 2 * pw), dtype=np.float64)
    padded[ph:ph + H, pw:pw + W] = img

    Kflip = K[::-1, ::-1]

    out = np.zeros((H, W), dtype=np.float64)
    for m in range(kh):
        for n in range(kw):
            w = Kflip[m, n]
            if w == 0.0:
                continue
            out += w * padded[m:m + H, n:n + W]
    return out


def conv2d_im2col(img, K):
    """TODO 1.1: build the (H*W, kh*kw) patch matrix, then one matmul.
    numpy.lib.stride_tricks.sliding_window_view is allowed. Report the peak
    memory and be ready to explain which step actually costs it."""
    H, W = img.shape
    kh, kw = K.shape
    ph, pw = kh // 2, kw // 2
    padded = np.zeros((H + 2 * ph, W + 2 * pw), dtype=np.float64)
    padded[ph:ph + H, pw:pw + W] = img

    Kflip = K[::-1, ::-1]

    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    # windows has shape (H, W, kh, kw): windows[i, j] is the kh*kw patch
    # feeding output pixel (i, j).
    patches = windows.reshape(H * W, kh * kw)
    out = patches @ Kflip.reshape(kh * kw)
    return out.reshape(H, W)


def conv2d_fft(img, K):
    """TODO 1.1: multiply in the frequency domain. numpy.fft is allowed."""
    H, W = img.shape
    kh, kw = K.shape
    ph, pw = kh // 2, kw // 2

    # Linear (not circular) convolution needs a common canvas big enough
    # for the full convolution, H+kh-1 by W+kw-1, or FFT wraparound will
    # corrupt the borders.
    fh, fw = H + kh - 1, W + kw - 1

    img_f = np.fft.rfft2(img, s=(fh, fw))
    K_f = np.fft.rfft2(K, s=(fh, fw))
    full = np.fft.irfft2(img_f * K_f, s=(fh, fw))

    # 'full' output is size (H+kh-1, W+kw-1); crop the centered HxW window
    # to match 'same' mode (same convention as the zero-padding above).
    return full[ph:ph + H, pw:pw + W]


def _conv1d_along_axis(img, tap, axis):
    """1-D convolution (zero-padded, 'same', true convolution i.e. tap is
    already flipped by the caller) of img with a 1-D vector, along axis 0
    (columns, vertical) or axis 1 (rows, horizontal). Vectorised: loops only
    over the k taps, each tap contributes a shifted full-image slice."""
    k = tap.shape[0]
    p = k // 2
    if axis == 0:
        H, W = img.shape
        padded = np.zeros((H + 2 * p, W), dtype=np.float64)
        padded[p:p + H, :] = img
        out = np.zeros((H, W), dtype=np.float64)
        for m in range(k):
            w = tap[m]
            if w == 0.0:
                continue
            out += w * padded[m:m + H, :]
    else:
        H, W = img.shape
        padded = np.zeros((H, W + 2 * p), dtype=np.float64)
        padded[:, p:p + W] = img
        out = np.zeros((H, W), dtype=np.float64)
        for n in range(k):
            w = tap[n]
            if w == 0.0:
                continue
            out += w * padded[:, n:n + W]
    return out


def conv2d_separable(img, K, tol=1e-10):
    """TODO 1.3: rank-1 only. Raise if K is not rank-1.

    K = sigma_1 * u_1 v_1^T (rank-1 factorisation). True 2-D convolution
    with K equals convolving with the column vector u_1 along one axis and
    the row vector v_1 along the other (in either order), scaled by
    sigma_1. Convolution (not correlation) means the SEPARABLE FACTORS
    themselves must be flipped before use, since convolving with each
    flipped 1-D factor in turn is equivalent to convolving with the fully
    flipped 2-D kernel K[::-1, ::-1] = sigma_1 * u_1[::-1] v_1[::-1]^T.
    """
    U, S, Vt = np.linalg.svd(K)
    rank, _, _ = numeric_rank(K, tol=tol)
    if rank > 1:
        raise ValueError(f"conv2d_separable: kernel is not rank-1 (numeric rank {rank})")

    sigma1 = S[0]
    u1 = U[:, 0]          # column factor, length kh
    v1 = Vt[0, :]         # row factor, length kw

    u1_flipped = u1[::-1]
    v1_flipped = v1[::-1]

    out = _conv1d_along_axis(img, u1_flipped, axis=0)
    out = _conv1d_along_axis(out, v1_flipped, axis=1)
    return sigma1 * out


def conv2d_lowrank(img, K, r):
    """TODO 1.3: sum of r separable passes from the truncated SVD.

    K ~= sum_{i=1}^r sigma_i u_i v_i^T. Each rank-1 term is applied with two
    1-D convolutions exactly as in conv2d_separable, and the r results are
    summed. Cost is O(2 r H W k) versus O(H W k^2) for full 2-D convolution.
    """
    U, S, Vt = np.linalg.svd(K)
    r = min(r, S.shape[0])

    out = np.zeros_like(img, dtype=np.float64)
    for i in range(r):
        sigma_i = S[i]
        if sigma_i == 0.0:
            continue
        ui_flipped = U[:, i][::-1]
        vi_flipped = Vt[i, :][::-1]
        term = _conv1d_along_axis(img, ui_flipped, axis=0)
        term = _conv1d_along_axis(term, vi_flipped, axis=1)
        out += sigma_i * term
    return out


def psnr(a, b, peak=255.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)


if __name__ == "__main__":
    from pathlib import Path
    from PIL import Image
    from scipy.signal import convolve2d          # checking only
    root = Path(__file__).resolve().parents[2]
    img = np.asarray(Image.open(root / "images/p1/base_2048.png")).astype(float)[:128, :128]
    K = kernel_bank(7)["gaussian"]
    ref = convolve2d(img, K, mode="same", boundary="fill")
    for fn in (conv2d_loops, conv2d_taps, conv2d_im2col, conv2d_fft):
        try:
            print(f"{fn.__name__:16s} max|err| = {np.abs(fn(img, K) - ref).max():.3e}")
        except NotImplementedError:
            print(f"{fn.__name__:16s} not implemented")
