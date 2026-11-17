"""Held-out check on RESIDE: no parameter in this script was tuned on RESIDE.

Runs the same methods and metrics as evaluate_dehazing.py, with the settings
chosen on NH-HAZE and O-HAZE, and writes tables only (no images).

    python evaluate_heldout.py
"""
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dehaze import process

DATASET = Path("datasets/datasets_quarter_size/RESIDE")
RESULTS_ROOT = Path("evaluation_results/reside_heldout")
CATEGORIES = ["indoor", "outdoor"]
METHODS = ["original", "dcp", "fixed", "matched_fixed", "adaptive"]

# Same DCP settings as the NH-HAZE / O-HAZE experiments.
COMMON = dict(patch=15, omega=0.90, t_min=0.1, radius=20, eps=1e-3, alpha=0.5,
              saturation_coef=-1.0, contrast_coef=-1.0, gradient_coef=-0.5,
              bias=-0.5, k=1.0)
# b1_5 = initial heuristic; b3 = best brightness coefficient on NH-HAZE / O-HAZE.
CONFIGS = {"b1_5": 1.5, "b3": 3.0}


def load_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def evaluate(name, brightness_coef):
    out = RESULTS_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    config = dict(COMMON, brightness_coef=brightness_coef, dataset="RESIDE",
                  resolution="quarter")
    (out / "config.json").write_text(json.dumps(config, indent=4) + "\n")
    rows = []
    for category in CATEGORIES:
        for hazy_path in sorted((DATASET / "hazy" / category).glob("*.jpg")):
            gt_path = (DATASET / "GT" / category
                       / hazy_path.name.replace("_hazy_", "_GT_", 1))
            hazy, gt = load_rgb(hazy_path), load_rgb(gt_path)
            results = process(hazy, brightness_coef=brightness_coef, **COMMON)
            row = dict(category=category, image=hazy_path.name,
                       weight_mean=float(results["weights"].mean()),
                       weight_std=float(results["weights"].std()))
            for method in METHODS:
                row[f"{method}_psnr"] = float(peak_signal_noise_ratio(
                    gt, results[method], data_range=1.0))
                row[f"{method}_ssim"] = float(structural_similarity(
                    gt, results[method], data_range=1.0, channel_axis=2))
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "all_image_metrics.csv", index=False)
    summary = []
    for category, group in frame.groupby("category"):
        for method in METHODS:
            summary.append(dict(
                category=category, method=method,
                mean_psnr=group[f"{method}_psnr"].mean(),
                mean_ssim=group[f"{method}_ssim"].mean(),
                image_count=len(group)))
        summary.append(dict(
            category=category, method="adaptive_minus_matched_fixed",
            mean_psnr=(group.adaptive_psnr - group.matched_fixed_psnr).mean(),
            mean_ssim=(group.adaptive_ssim - group.matched_fixed_ssim).mean(),
            image_count=len(group)))
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "method_comparison.csv", index=False)
    print(name)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    for config_name, coef in CONFIGS.items():
        evaluate(config_name, coef)
