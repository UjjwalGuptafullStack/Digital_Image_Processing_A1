# Digital Image Processing — From Scratch

All code, notebooks, images, and the report are on GitHub:
https://github.com/UjjwalGuptafullStack/Digital_Image_Processing_A1

The same `Images/` folder (every input asset and every generated output/figure) is
also mirrored on OneDrive, for anyone who would rather browse it without cloning the
repo:
https://iiithydresearch-my.sharepoint.com/:f:/g/personal/ujjwal_gupta_research_iiit_ac_in/IgAv3--vgP-eSIlLIUivK72DASPb_zGfvKQBRvsiM9eGRzI?e=AkKHOW

A collection of core image-processing algorithms implemented from first principles in
NumPy — no `scipy.ndimage`, `scipy.signal`, `cv2`, or `skimage` calls in the actual
implementations (only used to cross-check results). Covers 2-D convolution, bit-plane
steganography, piecewise-linear tone curves, histogram-based contrast methods, and
green-screen video compositing.

Each topic has a runnable Jupyter notebook that regenerates every figure and result
directly from the source images, so every number in `report.pdf` traces back to code
you can re-run.

## What's implemented

- **Convolution** — four implementations of 2-D convolution (naive nested loops,
  vectorised tap-loop, im2col, FFT) verified against each other and against
  `scipy.signal.convolve2d`; kernel rank analysis via SVD; low-rank/separable
  convolution approximation; runtime and memory scaling benchmarks.
- **Bit-plane steganography** — bit-plane decomposition, LSB payload embedding and
  extraction with a length-prefixed header format, and a custom block-mean
  quantisation-index-modulation scheme that survives Gaussian noise and JPEG
  compression where naive LSB embedding fails outright.
- **Piecewise-linear tone curves** — recovering an unknown pointwise transform from
  input/output image pairs via dynamic-programming breakpoint search, with automatic
  segment-count selection (BIC with a quantisation-noise floor) and an analysis of
  when a transform is identifiable from the data at all.
- **Histogram processing** — global equalisation (with proofs of idempotence and
  non-flatness), histogram matching/specification via inverse-CDF composition,
  hue-preserving colour equalisation, CLAHE implemented from scratch (tiled,
  clip-limited, bilinearly interpolated), and a parameter-free auto-exposure
  corrector.
- **Green-screen compositing** — chroma-distance keying, soft alpha mattes, spill
  suppression, and a vectorised video compositing pipeline with temporal
  flicker reduction.

## Directory structure

```
.
├── code/                      Python implementations, one folder per topic
│   ├── p1_convolution/
│   │   ├── starter.py         Core implementations (conv2d_*, numeric_rank, ...)
│   │   └── run_*.ipynb        Notebooks that run the analysis and display figures/results
│   ├── p2_bitplane/
│   │   ├── starter.py         Bit-plane ops, LSB embed/extract, robust embedding
│   │   └── run_*.ipynb
│   ├── p3_plt/
│   │   ├── starter.py         Transfer-curve recovery, piecewise fitting, identifiability
│   │   └── run_*.ipynb
│   ├── p4_histogram/
│   │   ├── starter.py         Equalisation, specification, CLAHE, auto-correction
│   │   └── run_*.ipynb
│   ├── p5_greenscreen/
│   │   ├── starter.py         Keying, matting, spill suppression, compositing
│   │   └── run_*.ipynb
│   └── assets/
│       ├── ingest_greenscreen.py   Extracts a usable 48-frame window from a video clip
│       └── check_asset.py          Sanity-checks extracted footage for keying quality
│
├── Images/
│   ├── input/                 Every source asset the notebooks read, by problem
│   │   ├── P1/                 base_2048.png
│   │   ├── P2/                 cover_textured.png, cover_smooth.png, decode_me.png
│   │   ├── P3/                 field_in/out, nebula_in/out, field_skycrop_in/out
│   │   ├── P4/                 colour_source.png, local_regions.png, equalization/,
│   │   │                       exposure/, matching/, specification/
│   │   └── P5/                 green_screen_clip.mp4, pano.jpg
│   └── output/                 Every figure/result the notebooks generate, by problem
│       ├── P1/                 conv2d error heatmaps, rank spectra, PSNR-vs-r,
│       │                       runtime plots (from run_1_1..run_1_4)
│       ├── P2/                 bit planes, PSNR curves, payload recovery, BER plots
│       ├── P3/                 transform-recovery scatter/fit plots, BIC selection
│       ├── P4/                 equalisation, matching/specification, CLAHE ablation,
│       │                       auto-exposure correction
│       └── P5/                 naive/soft keying, spill suppression, composite_raw.mp4
│
├── report.pdf                 Full write-up: one section per topic
├── ATTRIBUTION.md             Source and licence for every non-original image/clip
├── requirements.txt           Python dependencies
└── DIP_A1.pdf                 Original problem specification
```

Every `code/*/starter.py` holds the actual from-scratch algorithm implementations.
Every `code/*/run_*.ipynb` is a notebook — open it and run all cells to reproduce the
corresponding figures and printed results directly from the images in `Images/input/`;
nothing is written back to disk, so re-running a notebook never touches the
repository's tracked files (`Images/output/composite_raw.mp4` is the one exception —
5.3 writes its rendered composite video there).

`Images/output/` files are named `<notebook>_figNN.png`, in the order each figure
appears when the notebook is run top to bottom — e.g. `run_4_3_local_fig03.png` is the
third figure produced by `code/p4_histogram/run_4_3_local.ipynb`. These are exactly
the same images already embedded as cell outputs inside the notebooks; this folder
just makes them browsable without opening Jupyter.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Open any `code/*/run_*.ipynb` and run all cells. Notebooks assume they're run with
their own folder as the working directory (Jupyter's default when you open a notebook
from that folder), so that paths like `../../Images/input/P2/decode_me.png` resolve
correctly.

See `ATTRIBUTION.md` for the licence and source of every image or video not
generated by the code in this repository.
