
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from dehaze import process, save_rgb

import json


# ============================================================
# CONFIGURATION
# ============================================================

DATASETS_ROOT = Path("datasets/datasets_quarter_size")
RESULTS_ROOT = Path("evaluation_results/feature_ablation/no_bias")

DATASETS = [
    "NH-HAZE",
    "O-HAZE"
]

CATEGORIES = ["outdoor"]

METHODS = [
    "original",
    "dcp",
    "fixed",
    "matched_fixed",
    "adaptive"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# DCP and blending parameters
PATCH = 15
OMEGA = 0.90
T_MIN = 0.1
RADIUS = 20
EPS = 1e-3
ALPHA = 0.5

BRIGHTNESS_COEF = 1.5,
SATURATION_COEF = -1.0,
CONTRAST_COEF = -1.0,
GRADIENT_COEF = -0.5,
BIAS = 0.0

# Set False to save disk space
SAVE_IMAGES = True

RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

config = {
    "datasets": DATASETS,
    "resolution": "quarter",
    "patch": PATCH,
    "omega": OMEGA,
    "t_min": T_MIN,
    "radius": RADIUS,
    "eps": EPS,
    "alpha": ALPHA,
    "brightness_coef": BRIGHTNESS_COEF,
    "saturation_coef": SATURATION_COEF,
    "contrast_coef": CONTRAST_COEF,
    "gradient_coef": GRADIENT_COEF,
    "bias": BIAS
}

with open(RESULTS_ROOT / "config.json", "w") as f:
    json.dump(config, f, indent=4)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_rgb(path):
    """
    Load an image as RGB float32 in [0, 1].
    """

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not load: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image.astype(np.float32) / 255.0


# ============================================================
# FIND HAZY / GROUND-TRUTH PAIRS
# ============================================================

def find_image_pairs(dataset_name):
    """
    Find matching hazy and ground-truth images.

    Expected directory structure:

    datasets/
        NH-HAZE/
            hazy/outdoor/
            GT/outdoor/
        O-HAZE/
            hazy/outdoor/
            GT/outdoor/
        RESIDE/
            hazy/indoor/
            hazy/outdoor/
            GT/indoor/
            GT/outdoor/

    Expected filenames:

        0001_hazy_outdoor_ohaze.jpg
        0001_GT_outdoor_ohaze.jpg

    Image extensions may differ.
    """

    dataset_dir = DATASETS_ROOT / dataset_name

    pairs = []

    for category in CATEGORIES:

        hazy_dir = dataset_dir / "hazy" / category
        gt_dir = dataset_dir / "GT" / category

        if not hazy_dir.exists():
            continue

        if not gt_dir.exists():
            raise FileNotFoundError(
                f"Missing ground-truth directory: {gt_dir}"
            )

        gt_files = {
            path.stem.lower(): path
            for path in gt_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        }

        hazy_files = sorted(
            path
            for path in hazy_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for hazy_path in hazy_files:

            # Convert hazy filename to expected GT filename
            gt_stem = hazy_path.stem.lower().replace(
                "_hazy_", "_gt_", 1
            )

            gt_path = gt_files.get(gt_stem)

            if gt_path is None:
                raise FileNotFoundError(
                    f"No matching ground truth for: {hazy_path}\n"
                    f"Expected GT stem: {gt_stem}"
                )

            pairs.append({
                "dataset": dataset_name,
                "category": category,
                "image": hazy_path.name,
                "hazy_path": hazy_path,
                "gt_path": gt_path
            })

    if not pairs:
        raise ValueError(
            f"No image pairs found for {dataset_name}"
        )

    return pairs


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(ground_truth, predicted):
    """
    Calculate PSNR and color SSIM.
    Both images must be RGB float32 in [0, 1].
    """

    if ground_truth.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"{ground_truth.shape} vs {predicted.shape}"
        )

    psnr = peak_signal_noise_ratio(
        ground_truth,
        predicted,
        data_range=1.0
    )

    ssim = structural_similarity(
        ground_truth,
        predicted,
        data_range=1.0,
        channel_axis=2
    )

    return float(psnr), float(ssim)


# ============================================================
# EVALUATE ONE DATASET
# ============================================================

def evaluate_dataset(dataset_name):

    pairs = find_image_pairs(dataset_name)

    output_dir = RESULTS_ROOT / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []

    print("\n" + "=" * 70)
    print(f"Evaluating dataset: {dataset_name}")
    print(f"Number of image pairs: {len(pairs)}")
    print("=" * 70)

    for index, pair in enumerate(pairs, start=1):

        hazy_path = pair["hazy_path"]
        gt_path = pair["gt_path"]

        # ----------------------------------------------------
        # Load images
        # ----------------------------------------------------

        hazy = load_rgb(hazy_path)
        ground_truth = load_rgb(gt_path)

        if hazy.shape != ground_truth.shape:
            raise ValueError(
                f"Shape mismatch for {hazy_path.name}: "
                f"{hazy.shape} vs {ground_truth.shape}"
            )

        # ----------------------------------------------------
        # Run dehazing methods
        # ----------------------------------------------------

        results = process(
            hazy,
            patch=PATCH,
            omega=OMEGA,
            t_min=T_MIN,
            radius=RADIUS,
            eps=EPS,
            alpha=ALPHA,
            brightness_coef=BRIGHTNESS_COEF,
            saturation_coef=SATURATION_COEF,
            contrast_coef= CONTRAST_COEF,
            gradient_coef=GRADIENT_COEF,
            bias=BIAS
        )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        metrics = {}

        metrics["weight_mean"] = float(np.mean(results["weights"]))
        metrics["weight_std"] = float(np.std(results["weights"]))
        metrics["weight_min"] = float(np.min(results["weights"]))
        metrics["weight_max"] = float(np.max(results["weights"]))

        for method in METHODS:

            psnr, ssim = calculate_metrics(
                ground_truth,
                results[method]
            )

            metrics[f"{method}_psnr"] = psnr
            metrics[f"{method}_ssim"] = ssim

        # Improvements over original hazy image
        for method in ["dcp", "fixed", "adaptive"]:

            metrics[f"{method}_psnr_gain"] = (
                metrics[f"{method}_psnr"]
                - metrics["original_psnr"]
            )

            metrics[f"{method}_ssim_gain"] = (
                metrics[f"{method}_ssim"]
                - metrics["original_ssim"]
            )

        # Adaptive vs DCP
        metrics["adaptive_vs_dcp_psnr"] = (
            metrics["adaptive_psnr"]
            - metrics["dcp_psnr"]
        )

        metrics["adaptive_vs_dcp_ssim"] = (
            metrics["adaptive_ssim"]
            - metrics["dcp_ssim"]
        )

        # Adaptive vs fixed
        metrics["adaptive_vs_fixed_psnr"] = (
            metrics["adaptive_psnr"]
            - metrics["fixed_psnr"]
        )

        metrics["adaptive_vs_fixed_ssim"] = (
            metrics["adaptive_ssim"]
            - metrics["fixed_ssim"]
        )

        metrics["adaptive_vs_matched_psnr"] = (
                metrics["adaptive_psnr"]
                - metrics["matched_fixed_psnr"]
        )

        metrics["adaptive_vs_matched_ssim"] = (
                metrics["adaptive_ssim"]
                - metrics["matched_fixed_ssim"]
        )

        # ----------------------------------------------------
        # Save processed images
        # ----------------------------------------------------

        if SAVE_IMAGES:

            image_dir = (
                output_dir
                / "images"
                / pair["category"]
                / hazy_path.stem
            )

            image_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            for method in METHODS:

                save_rgb(
                    image_dir / f"{method}.png",
                    results[method]
                )

            save_rgb(
                image_dir / "ground_truth.png",
                ground_truth
            )

            save_rgb(
                image_dir / "weights.png",
                results["weights"]
            )

            save_rgb(
                image_dir / "transmission.png",
                results["transmission"]
            )

        # ----------------------------------------------------
        # Record results
        # ----------------------------------------------------

        records.append({
            "dataset": dataset_name,
            "category": pair["category"],
            "image": pair["image"],
            "hazy_path": str(hazy_path),
            "gt_path": str(gt_path),
            **metrics
        })

        print(
            f"[{index}/{len(pairs)}] "
            f"{pair['image']} | "
            f"DCP: {metrics['dcp_psnr']:.2f} dB | "
            f"Fixed: {metrics['fixed_psnr']:.2f} dB | "
            f"Adaptive: {metrics['adaptive_psnr']:.2f} dB"
        )

    results_df = pd.DataFrame(records)

    results_df.to_csv(
        output_dir / "image_metrics.csv",
        index=False
    )

    return results_df


# ============================================================
# CREATE SUMMARY TABLES
# ============================================================

def create_summary(all_results):

    metric_columns = [
        column
        for column in all_results.columns
        if column.endswith("_psnr")
        or column.endswith("_ssim")
        or column.endswith("_gain")
        or column.startswith("adaptive_vs_")
    ]

    # Mean metrics per dataset
    dataset_summary = (
        all_results
        .groupby("dataset")[metric_columns]
        .mean()
        .reindex(DATASETS)
    )

    dataset_summary.to_csv(
        RESULTS_ROOT / "dataset_comparison.csv"
    )

    # Mean metrics by dataset and category
    category_summary = (
        all_results
        .groupby(["dataset", "category"])[metric_columns]
        .mean()
    )

    category_summary.to_csv(
        RESULTS_ROOT / "category_comparison.csv"
    )

    # Overall averages
    overall_summary = (
        all_results[metric_columns]
        .mean()
        .to_frame(name="mean")
    )

    overall_summary.to_csv(
        RESULTS_ROOT / "overall_summary.csv"
    )

    # Compact readable comparison
    rows = []

    for dataset in DATASETS:

        dataset_df = all_results[
            all_results["dataset"] == dataset
        ]

        for method in METHODS:

            rows.append({
                "dataset": dataset,
                "method": method,
                "mean_psnr": dataset_df[
                    f"{method}_psnr"
                ].mean(),
                "mean_ssim": dataset_df[
                    f"{method}_ssim"
                ].mean(),
                "image_count": len(dataset_df)
            })

    comparison = pd.DataFrame(rows)

    comparison.to_csv(
        RESULTS_ROOT / "method_comparison.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("METHOD COMPARISON")
    print("=" * 70)

    print(
        comparison.round(4).to_string(index=False)
    )

    return comparison


# ============================================================
# CREATE COMPARISON GRAPHS
# ============================================================

def plot_comparison(comparison, metric):

    pivot = comparison.pivot(
        index="dataset",
        columns="method",
        values=f"mean_{metric}"
    )

    pivot = pivot.reindex(
        index=DATASETS,
        columns=METHODS
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(11, 6),
        width=0.8
    )

    ax.set_title(
        f"Comparison of Dehazing Methods by Dataset ({metric.upper()})"
    )

    ax.set_xlabel("Dataset")

    if metric == "psnr":
        ax.set_ylabel("Mean PSNR (dB)")
    else:
        ax.set_ylabel("Mean SSIM")

    ax.tick_params(axis="x", rotation=0)

    ax.legend(
        title="Method",
        loc="best"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_ROOT / f"{metric}_comparison.png",
        dpi=200
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    all_results = []

    for dataset_name in DATASETS:

        dataset_results = evaluate_dataset(
            dataset_name
        )

        all_results.append(dataset_results)

    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    combined.to_csv(
        RESULTS_ROOT / "all_image_metrics.csv",
        index=False
    )

    comparison = create_summary(combined)

    plot_comparison(comparison, "psnr")
    plot_comparison(comparison, "ssim")

    print("\nEvaluation complete.")
    print(f"Results saved to: {RESULTS_ROOT.resolve()}")


if __name__ == "__main__":
    main()
