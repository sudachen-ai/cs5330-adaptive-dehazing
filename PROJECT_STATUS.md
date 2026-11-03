# Project status and next work

## Observed validation results

Default parameters, quarter-size images, 79 validation pairs; RGB float [0,1],
mean per-image PSNR / color SSIM. No parameters were optimized in this run.

| Dataset | n | DCP | Fixed alpha=0.5 | Matched fixed | Adaptive |
|---|---:|---:|---:|---:|---:|
| NH-HAZE | 43 | 12.8176 / .5019 | 13.2252 / .4949 | 13.2614 / .4974 | 13.3327 / .4940 |
| O-HAZE | 36 | 16.0401 / .6930 | 17.4933 / .7197 | 17.5295 / .7215 | 17.5171 / .7182 |

Adaptive PSNR exceeds fixed alpha=0.5 by only 0.1075 / 0.0238 dB,
while SSIM is lower on both datasets. Against matched fixed, adaptive PSNR
increases by 0.0713 dB on NH-HAZE and decreases by 0.0124 dB on O-HAZE.
These results do not establish a robust benefit from spatial adaptation.
Per-image values and plots are in `reports/validation_default`.

## Proposed division of work (confirm with teammate)

The shared chat says the teammate is preparing baseline experiments/results;
it does not establish a final division of responsibility or course rubric.
Avoid duplicating their ongoing runs.

- Teammate: upload existing configurations/results and describe dataset provenance,
  previous tuning, and baseline conclusions.
- Suda: lead validation-only coefficient/patch experiments, compare to a tuned
  global alpha, inspect failure cases, and draft the adaptive-method discussion.
- Together: agree on frozen parameters, evaluate held-out scenes once, assemble
  final report and presentation, and verify all course deliverables/deadlines.

## Experiment checklist

1. Confirm dataset sources, licenses, pairing/alignment, scene IDs, and whether
   the proposed test scenes have already influenced decisions. Dataset provenance
   and redistribution permission are not documented by this change.
2. On validation only, sweep global alpha (e.g. 0,.25,.5,.75,1), then a small
   predeclared grid for adaptive bias/coefficients and patch size. Choose one
   objective and aggregation rule before selecting the winner. Use identical DCP
   parameters and resolution across methods.
3. Ablate brightness, saturation, contrast, gradient (set each coefficient to 0).
   Add an explicit raw-weight/no-refinement variant before claiming that ablation.
4. Show sky/white surfaces, texture, haze variation, halos, and color shifts.
   Include both wins and failures and display GT, DCP, tuned fixed, adaptive,
   and weights with the same rendering scale.
5. Freeze parameters and evaluate `splits/real_haze/test.csv` in a fresh output
   directory. Report per-dataset sample counts, PSNR/SSIM, paired differences,
   variability, runtime, and limitations. Never select settings from test scores.
6. Complete report/slides with method equations, references, results, limitations,
   individual contributions, and any additional course-required material.

The assignment specification and earlier proposed schedule were not supplied;
this checklist is based on repository evidence, not a claim of rubric compliance.
