"""P2 starter. Implement every function marked TODO."""
import numpy as np


def bit_planes(img):
    """TODO 2.1: (H,W) uint8 -> (8,H,W) uint8 in {0,1}, index 0 = LSB."""
    img = img.astype(np.uint8)
    return np.stack([(img >> b) & 1 for b in range(8)], axis=0)


def reconstruct(planes, keep):
    """TODO 2.1: rebuild from the `keep` most significant planes.

    `keep` MSB planes means bit indices 7, 6, ..., 7-keep+1 (indices 0=LSB..7=MSB
    as produced by bit_planes). The remaining (unused) low bits are treated as 0.
    """
    out = np.zeros(planes.shape[1:], dtype=np.uint16)
    for b in range(8 - keep, 8):
        out |= (planes[b].astype(np.uint16) << b)
    return out.astype(np.uint8)


def gray_encode(img):
    """TODO 2.1: Gray-coded intensities, for the ramp comparison.

    Standard binary-to-Gray: gray = img XOR (img >> 1). Adjacent integers
    differ by exactly one bit in the Gray code, unlike standard binary.
    """
    img = img.astype(np.uint8)
    return img ^ (img >> 1)


def embed_lsb(cover, bits, plane=0):
    """TODO 2.2: write bits into the given plane, raster order.

    `bits` is a 1-D array/sequence of 0/1 values, written into pixels in
    raster order (row-major, top-left first). len(bits) must not exceed
    cover.size. Every position in range gets overwritten, including
    positions whose original bit already equalled the value being written.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    H, W = cover.shape
    flat = cover.astype(np.uint8).flatten()
    mask = np.uint8(~(1 << plane) & 0xFF)
    flat[:len(bits)] = (flat[:len(bits)] & mask) | (bits << plane)
    return flat.reshape(H, W)


def extract_lsb(stego, n, plane=0):
    """TODO 2.2: read n bits back out.

    Reads the given bit-plane of the first n pixels in raster order.
    """
    flat = stego.astype(np.uint8).flatten()
    return (flat[:n] >> plane) & 1


def embed_robust(cover, bits, block_size=16, delta=2.0, **kw):
    """TODO 2.3: block-mean QIM with full-image majority-vote repetition.

    Design: partition the image into non-overlapping block_size x block_size
    blocks (1024/16 = 64 per side = 4096 blocks total for a 1024x1024
    cover). Each of the len(bits) message bits is assigned a disjoint,
    contiguous run of `reps = total_blocks // len(bits)` blocks (128 bits ->
    32 blocks/bit, using every block in the image). For each assigned
    block, every pixel in that block is shifted by +delta (bit=1) or -delta
    (bit=0), clipped to [0, 255].

    Why block means, not individual-pixel LSBs: averaging over a
    block_size^2-pixel block divides the effective noise standard
    deviation seen by the block mean by block_size (CLT), so additive
    Gaussian noise that would swamp a single-pixel LSB by two orders of
    magnitude has almost no effect on the block mean. JPEG quantises each
    8x8 block's DC coefficient (proportional to the block mean) far more
    finely than its AC/high-frequency coefficients, so a block-mean shift
    survives JPEG compression that would destroy single-pixel changes.
    Repeating each bit across 32 independent blocks and majority-voting at
    decode adds a second, independent layer of robustness on top of the
    per-block averaging.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n_bits = len(bits)
    H, W = cover.shape
    bpr, bpc = W // block_size, H // block_size
    total_blocks = bpr * bpc
    reps = total_blocks // n_bits
    if reps < 1:
        raise ValueError("not enough blocks to embed this many bits")

    stego = cover.astype(np.float64).copy()
    for bit_i in range(n_bits):
        shift = delta if bits[bit_i] == 1 else -delta
        for rep in range(reps):
            idx = bit_i * reps + rep
            by, bx = divmod(idx, bpr)
            y0, x0 = by * block_size, bx * block_size
            stego[y0:y0 + block_size, x0:x0 + block_size] += shift
    return np.clip(np.round(stego), 0, 255).astype(np.uint8)


def extract_robust(stego, cover, n, block_size=16, **kw):
    """TODO 2.3: matching extractor (cover-assisted).

    For each of the `reps` blocks assigned to bit i, compares the block's
    mean in `stego` to its mean in the clean `cover`: a positive difference
    votes for bit 1, negative votes for bit 0. Majority vote across the
    `reps` blocks gives the final bit. Needs the clean cover to know the
    pre-embedding block means (see the design discussion for what a blind
    variant would require instead).
    """
    H, W = stego.shape
    bpr, bpc = W // block_size, H // block_size
    total_blocks = bpr * bpc
    reps = total_blocks // n

    stego_f = stego.astype(np.float64)
    cover_f = cover.astype(np.float64)
    out = np.zeros(n, dtype=np.uint8)
    for bit_i in range(n):
        votes = 0
        for rep in range(reps):
            idx = bit_i * reps + rep
            by, bx = divmod(idx, bpr)
            y0, x0 = by * block_size, bx * block_size
            s_mean = stego_f[y0:y0 + block_size, x0:x0 + block_size].mean()
            c_mean = cover_f[y0:y0 + block_size, x0:x0 + block_size].mean()
            votes += 1 if (s_mean - c_mean) > 0 else -1
        out[bit_i] = 1 if votes > 0 else 0
    return out


# ---- provided channels: do not modify, these are what you are graded against
def degrade_gaussian(img, sigma, rng=None):
    rng = rng or np.random.default_rng(0)
    return np.clip(np.round(img.astype(np.float64) + rng.normal(0, sigma, img.shape)),
                   0, 255).astype(np.uint8)


def degrade_jpeg(img, quality):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L"))


def ber(a, b):
    return float(np.mean(a != b))


def psnr(a, b, peak=255.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)
