# Why adaptive blending fails on RESIDE

Script: `analyze_reside_failure.py` (run from the repository root, about 30 s on
a laptop CPU). Settings are the same as `evaluate_heldout.py` with brightness
coefficient 3.0. Outputs: `summary.csv`, `per_image.csv`,
`reside_outdoor_examples.png`.

## 1. The loss is systematic, not a few outliers

On RESIDE outdoor, adaptive loses to `matched_fixed` on 489 of 492 images
(win rate 0.6%). On NH-HAZE it wins on 74% and on O-HAZE on 60%.

## 2. Exact decomposition of the gap

Write the weight as `w = m + d` (m = image mean, d = deviation),
`D = dcp - hazy`, `r = matched_fixed - gt`. Then exactly:

    MSE(adaptive) - MSE(matched_fixed) = 2·mean(d·r·D) + mean(d²·D²)
                                         alignment       variance penalty

The penalty is never negative. Spatial variation helps only if the alignment
term is negative, that is, if the weight is high where DCP is locally better.
The decomposition matches the measured gap to numerical precision.

| Set | n | Win rate | Gain (dB) | Alignment | Penalty | corr(w, oracle w) |
|---|---|---|---|---|---|---|
| NH-HAZE | 54 | 74% | +0.09 | -0.0013 | 0.0003 | +0.23 |
| O-HAZE | 45 | 60% | +0.01 | -0.0004 | 0.0002 | +0.12 |
| RESIDE indoor | 50 | 20% | -0.39 | +0.0008 | 0.0005 | -0.46 |
| RESIDE outdoor | 492 | 0.6% | -1.49 | +0.0036 | 0.0006 | -0.89 |

On RESIDE the alignment term is positive and about 6× the penalty: the weight
map is not merely noisy, it points the wrong way.

The "oracle weight" is the least-squares blend weight in each 15×15 window,
computed from the ground truth. It is an upper bound for diagnosis only, not
a method.

## 3. Cause: the brightness term means opposite things in the two data sources

Bright, flat regions (brightness > 0.7, gradient < 0.08; mostly sky and white
walls) cover 15–49% of the pixels:

| Set | Our weight, bright-flat | Oracle, bright-flat | Our weight, rest | Oracle, rest |
|---|---|---|---|---|
| NH-HAZE | 0.82 | 0.73 | 0.69 | 0.54 |
| O-HAZE | 0.81 | 0.56 | 0.69 | 0.60 |
| RESIDE indoor | 0.86 | 0.76 | 0.73 | 0.94 |
| RESIDE outdoor | 0.87 | **0.18** | 0.67 | **0.77** |

- In NH-HAZE and O-HAZE (real haze), bright regions are mostly dense haze, so
  "bright → more DCP" is roughly right.
- In RESIDE outdoor (synthetic haze on clear-day photos), bright flat regions
  are mostly sky. DCP over-saturates and shifts the sky color, so the best
  weight there is low. The ground below needs more DCP than our map gives it.
  Our map is close to the inverse of the oracle
  (see `reside_outdoor_examples.png`: white sky in "weight (ours)", black sky
  in "oracle weight").

## 4. Check: flip the sign of the brightness term

Diagnostic only (not a tuned result; RESIDE outdoor uses every 4th image):

| Set | Brightness 3.0 | 0.0 | -1.5 |
|---|---|---|---|
| NH-HAZE gain (dB) | **+0.09** | +0.00 | -0.04 |
| O-HAZE gain (dB) | **+0.01** | -0.03 | -0.05 |
| RESIDE indoor gain (dB) | -0.39 | +0.05 | **+0.05** |
| RESIDE outdoor gain (dB) | -1.53 | -0.18 | **+0.28** (wins 100%) |

The sign that helps one data source hurts the other. A hand-set linear score
on brightness cannot serve both. Note: absolute adaptive PSNR still drops with
a negative coefficient because the mean weight drops; the gains above are
relative to `matched_fixed` at the same mean weight.

## 5. What to say in the report

1. Spatial adaptivity has real potential: the oracle weight gains +2.5 to
   +12.6 dB over `matched_fixed` (upper bound, uses ground truth).
2. Our heuristic captures almost none of it, because local brightness is an
   ambiguous cue: dense haze and sky are both bright and flat.
3. The RESIDE failure is a distribution shift between real and synthetic haze,
   and it shows that tuning on NH-HAZE / O-HAZE alone overfits the cue.

Possible follow-ups (pick one if time allows):
- Add a sky cue (e.g. bright + low gradient + upper image position) that
  lowers the weight, or use the DCP transmission map as a feature.
- Learn the score coefficients by regressing onto the oracle weight on a
  training split, and test on held-out images.
