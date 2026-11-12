# CS5330: Spatially Adaptive Image Dehazing

A Dark Channel Prior (DCP) baseline plus local-feature blending of the original
and dehazed image, with a paired-image evaluation on NH-HAZE, O-HAZE, and RESIDE.

**Status:** pipeline and evaluation run end to end. On real paired images the
adaptive weight map gives almost no gain over a single fixed blend weight, and
it is worse on the held-out RESIDE set. See [Results](#results).

## Quick start (including VS Code)

Open this folder in VS Code with **File → Open Folder**, then open its terminal.
Python 3.10+ is recommended. No GPU is required.
The prototype was checked on macOS with Python 3.13, NumPy 2.5.4, and OpenCV
4.14.0.94. `requirements-lock.txt` records that exact dependency pair;
`requirements.txt` allows compatible versions.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py
python -m unittest -v
```

On Windows, activate with `.venv\Scripts\activate` instead.
In VS Code, use **Python: Select Interpreter** to select `.venv`.

Process an image you have permission to use:

```bash
python dehaze.py data/hazy.jpg --output results/example
```

Run the paired-image evaluation (needs the extra packages in `requirements.txt`):

```bash
python evaluate_dehazing.py   # NH-HAZE + O-HAZE; settings are constants at the top of the file
python evaluate_heldout.py    # RESIDE, tables only, about 5 minutes on a laptop CPU
```

Each run saves `original.png`, `dcp.png`, `fixed.png`, `adaptive.png`,
`weights.png`, `raw_weights.png`, `transmission.png`, the four feature maps,
`comparison.png`, and `run.json` with parameters. Images are processed at their
original resolution; large images can take more memory.

## Pipeline

1. Compute the dark channel using a local minimum over RGB channels and a 15×15 window.
2. Select atmospheric light from the brightest RGB pixel among the top 0.1% dark-channel pixels.
3. Estimate transmission with omega=0.95 (`dehaze.py` default; every run in
   `evaluation_results/` uses omega=0.90), refine with a grayscale guided filter,
   and recover the DCP image with transmission floored at 0.1.
4. Extract mean HSV value, mean saturation, luminance standard deviation, and
   mean Sobel gradient magnitude in densely overlapping windows.
5. Normalize contrast and gradient using fixed scales, then compute:

   `score = -0.5 + 1.5 * brightness - saturation - contrast - 0.5 * gradient`

   `weight_raw = sigmoid(score)`

6. Refine weights with the original grayscale image as guidance; clip to [0,1].
7. Blend: `adaptive = (1 - weight) * original + weight * dcp`.

The fixed baseline uses alpha=0.5 by default. Larger weights mean stronger use
of the DCP output. All calculations use float32 RGB in [0,1]. Guided filtering
uses radius=20 and epsilon=0.001 by default. This is a grayscale-guidance
implementation, not the full color guided filter.

## Example

![Synthetic pipeline comparison](examples/synthetic/comparison.png)

The image is generated procedurally by `demo.py`. The panels show original
synthetic haze, DCP, fixed blending, adaptive blending, weights, and estimated
transmission. Grayscale weights and transmission maps use black=0, white=1.
This checks that the pipeline runs; it is not a natural-image dataset result.

## Results

All numbers use the quarter-size images in `datasets/datasets_quarter_size`.
PSNR is in dB; higher PSNR and higher SSIM are better. Methods:

- `original`: the hazy input, no processing.
- `dcp`: the DCP output.
- `fixed`: blend with weight 0.5 everywhere.
- `matched_fixed`: blend with one weight per image, equal to the mean of that
  image's adaptive weight map. It uses the same total amount of DCP as
  `adaptive`, so `adaptive - matched_fixed` isolates the effect of varying the
  weight across the image.
- `adaptive`: blend with the per-pixel weight map.

### NH-HAZE (54 images) and O-HAZE (45 images), initial coefficients

Source: `evaluation_results/feature_ablation/baseline/method_comparison.csv`.

| Method | NH-HAZE PSNR | NH-HAZE SSIM | O-HAZE PSNR | O-HAZE SSIM |
|---|---|---|---|---|
| original | 11.63 | 0.430 | 13.80 | 0.605 |
| dcp | 13.15 | 0.508 | 16.79 | 0.708 |
| fixed | 13.24 | 0.491 | 17.11 | 0.698 |
| matched_fixed | 13.29 | 0.494 | 17.15 | 0.700 |
| adaptive | 13.36 | 0.490 | 17.16 | 0.697 |

Adaptive blending has the highest PSNR, but its margin over `matched_fixed` is
0.08 dB on NH-HAZE and 0.01 dB on O-HAZE, and its SSIM is lower than DCP on
both datasets.

### Feature ablation

Change in adaptive PSNR when one term is removed from the score
(`evaluation_results/feature_ablation/comparisons/changes_from_baseline.csv`):

| Removed term | NH-HAZE | O-HAZE |
|---|---|---|
| brightness | -0.61 | -1.18 |
| bias | +0.12 | +0.34 |
| saturation | +0.11 | +0.26 |
| contrast | +0.03 | +0.07 |
| gradient | +0.005 | +0.02 |

Brightness is the only term whose removal lowers PSNR. Removing any other term
raises PSNR slightly. Those terms are all negative, so removing one raises the
mean weight; `matched_fixed` rises by a similar amount in each case, which
points to the amount of DCP, not the spatial pattern, as the cause.

### Brightness coefficient and sigmoid strength

- Brightness coefficient 0 to 5 (`evaluation_results/brightness/comparison`):
  adaptive PSNR peaks at 3.0 (13.56 dB NH-HAZE, 17.60 dB O-HAZE). The mean
  weight rises from 0.30 to 0.87 over the sweep. `adaptive - matched_fixed`
  stays between -0.03 and +0.09 dB, and adaptive SSIM is below `matched_fixed`
  at every setting.
- Sigmoid strength k from 0.5 to 8 (`evaluation_results/sigmoid_strength/comparison`):
  a larger k makes the weight map vary more across the image. On NH-HAZE the
  adaptive margin grows to +0.13 dB at k=4. On O-HAZE it turns negative
  (-0.11 dB at k=4, -0.38 dB at k=8). SSIM falls on both datasets as k grows.
- Saturation coefficient (`evaluation_results/saturation`): graphs only; no
  summary table is committed yet.

### Held-out check on RESIDE

No setting was tuned on RESIDE. Source:
`evaluation_results/reside_heldout/*/method_comparison.csv`
(50 indoor and 492 outdoor images).

| Method | Indoor PSNR | Indoor SSIM | Outdoor PSNR | Outdoor SSIM |
|---|---|---|---|---|
| original | 9.54 | 0.612 | 15.94 | 0.813 |
| dcp | 19.54 | 0.808 | 18.35 | 0.890 |
| fixed | 14.21 | 0.727 | 20.54 | 0.888 |
| matched_fixed, brightness 1.5 | 15.12 | 0.741 | 20.57 | 0.890 |
| adaptive, brightness 1.5 | 14.96 | 0.732 | 19.34 | 0.873 |
| matched_fixed, brightness 3.0 | 17.64 | 0.777 | 20.42 | 0.901 |
| adaptive, brightness 3.0 | 17.25 | 0.762 | 18.94 | 0.876 |

The result from NH-HAZE and O-HAZE does not transfer. On RESIDE outdoor,
adaptive is 1.2 to 1.5 dB below `matched_fixed`. On RESIDE indoor, plain DCP is
best by a wide margin and every blend loses to it.

Why: the brightness term raises the weight in bright, flat regions. In
NH-HAZE and O-HAZE these are mostly dense haze, where DCP helps. In RESIDE
outdoor they are mostly sky, where DCP over-saturates, so the weight map is
close to the inverse of a ground-truth oracle weight (correlation -0.89).
Details, an exact decomposition of the PSNR gap, and example figures:
[analysis_reside/FINDINGS.md](analysis_reside/FINDINGS.md)
(`python analyze_reside_failure.py`, about 30 seconds).

## Known limitations

- The coefficient sweeps were scored on all 99 NH-HAZE and O-HAZE images.
  There is no separate validation split in use, so the tuned numbers for those
  two datasets are optimistic. `prepare_data.py` can create a split, but the
  evaluation does not read it yet.
- The saturation experiment committed about 3.8 GB of output images. The
  repository is slow to clone because of this.

## Next steps

- Lower the weight in sky regions (bright, flat, upper part of the image), or
  fit the score coefficients to the oracle weight on a training split. See
  `analysis_reside/FINDINGS.md`.
- Split NH-HAZE and O-HAZE into validation and test scenes, tune on validation
  only, and report the test numbers.
- Export a summary table for the saturation sweep.
- Pick qualitative examples (one success, one failure per dataset) for the
  final report.

Brightness and low texture can reflect scene content rather than haze. The
weight map is a heuristic blending preference, not a calibrated haze estimate.
Blending can soften a bad DCP result but cannot recover details absent from both
inputs.

## References

- He, Sun, Tang. *Single Image Haze Removal Using Dark Channel Prior*, CVPR 2009.
  [Author project page](https://people.csail.mit.edu/kaiming/cvpr09/)
- He, Sun, Tang. *Guided Image Filtering*, ECCV 2010.
  [Author paper](https://kaiminghe.github.io/publications/eccv10guidedfilter.pdf)

The Python implementation in this repository was written for this prototype
with AI assistance. The algorithms above are attributed to their original authors.
