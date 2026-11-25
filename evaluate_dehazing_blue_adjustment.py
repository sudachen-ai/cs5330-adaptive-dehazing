from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from skimage.color import rgb2lab, deltaE_ciede2000
from dehaze_with_blue_correction import process, save_rgb

import json


# ============================================================
# CONFIGURATION
# ============================================================

DATASETS_ROOT = Path("datasets/datasets_quarter_size")


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
    "adaptive",
    "adaptive_corrected"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# DCP and blending parameters
PATCH = 15
OMEGA = 0.90
T_MIN = 0.1
RADIUS = 20
EPS = 1e-3
ALPHA = 0.5

BRIGHTNESS_COEF = 5.0
SATURATION_COEF = -1.0
CONTRAST_COEF = -4.0
GRADIENT_COEF = -2.0
BIAS = -1.0
K = 1.0

# Blue tint correction parameters (change one at a time).
BRIGHTNESS_THRESHOLD = 0.0
SATURATION_THRESHOLD = 1
CORRECTION_STRENGTH = 1.0
BLUE_THRESHOLD = 0.0
CORRECTION_RADIUS = 30

# Include all correction settings in the output folder to avoid
# overwriting runs when changing different parameters.
RESULTS_ROOT = Path("evaluation_results/blue_correction/saturation_over_1") / (
    f"strength_{CORRECTION_STRENGTH:g}_bt_{BRIGHTNESS_THRESHOLD:g}"
    f"_st_{SATURATION_THRESHOLD:g}_blue_{BLUE_THRESHOLD:g}"
    f"_r_{CORRECTION_RADIUS}"
)

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
    "bias": BIAS,
    "k": K,
    "brightness_threshold": BRIGHTNESS_THRESHOLD,
    "saturation_threshold": SATURATION_THRESHOLD,
    "blue_threshold": BLUE_THRESHOLD,
    "correction_strength": CORRECTION_STRENGTH,
    "correction_radius": CORRECTION_RADIUS
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
            bias=BIAS,
            k=K,
            brightness_threshold=BRIGHTNESS_THRESHOLD,
            saturation_threshold=SATURATION_THRESHOLD,
            blue_threshold=BLUE_THRESHOLD,
            correction_strength=CORRECTION_STRENGTH,
            correction_radius=CORRECTION_RADIUS
        )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        metrics = {}

        metrics["weight_mean"] = float(np.mean(results["weights"]))
        metrics["weight_std"] = float(np.std(results["weights"]))
        metrics["weight_min"] = float(np.min(results["weights"]))
        metrics["weight_max"] = float(np.max(results["weights"]))

        correction = results["correction_map"]
        metrics["mean_blue_correction"] = float(correction.mean())
        metrics["max_blue_correction"] = float(correction.max())
        metrics["fraction_pixels_corrected"] = float((correction > 1e-6).mean())
        gt_lab = rgb2lab(ground_truth)

        for method in METHODS:

            psnr, ssim = calculate_metrics(
                ground_truth,
                results[method]
            )

            metrics[f"{method}_psnr"] = psnr
            metrics[f"{method}_ssim"] = ssim
            metrics[f"{method}_delta_e"] = float(
                deltaE_ciede2000(gt_lab, rgb2lab(results[method])).mean()
            )

        # Improvements over original hazy image
        for method in METHODS[1:]:

            metrics[f"{method}_psnr_gain"] = (
                metrics[f"{method}_psnr"]
                - metrics["original_psnr"]
            )

            metrics[f"{method}_ssim_gain"] = (
                metrics[f"{method}_ssim"]
                - metrics["original_ssim"]
            )

            metrics[f"{method}_delta_e_gain"] = (
                metrics["original_delta_e"] - metrics[f"{method}_delta_e"]
            )

        # Positive gain means improvement for all three metrics.
        for metric in ("psnr", "ssim", "delta_e"):
            before = metrics[f"adaptive_{metric}"]
            after = metrics[f"adaptive_corrected_{metric}"]
            gain = (before - after) if metric == "delta_e" else (after - before)
            if before == after:
                gain = 0.0
            metrics[f"corrected_vs_adaptive_{metric}"] = gain
            metrics[f"correction_win_{metric}"] = float(gain > 0)

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
            save_rgb(image_dir / "correction_map.png", correction)
            maximum = correction.max()
            normalized = correction / maximum if maximum > 0 else np.zeros_like(correction)
            save_rgb(image_dir / "correction_map_normalized.png", normalized)

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
            f"Adaptive: {metrics['adaptive_psnr']:.2f} dB | "
            f"Corrected: {metrics['adaptive_corrected_psnr']:.2f} dB | "
            f"Delta E gain: {metrics['corrected_vs_adaptive_delta_e']:+.3f}"
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
        or column.endswith("_delta_e")
        or column.startswith("correction_win_")
        or column in ("mean_blue_correction", "max_blue_correction", "fraction_pixels_corrected")
        or column.endswith("_gain")
        or column.startswith("adaptive_vs_")
        or column.startswith("weight_")
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
                "mean_delta_e": dataset_df[f"{method}_delta_e"].mean(),
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
    elif metric == "ssim":
        ax.set_ylabel("Mean SSIM")
    else:
        ax.set_ylabel("Mean CIEDE2000 (lower is better)")

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

    # Dedicated per-image weight statistics
    weight_columns = [
        "dataset", "category", "image",
        "weight_mean", "weight_std", "weight_min", "weight_max"
    ]
    combined[weight_columns].to_csv(
        RESULTS_ROOT / "weight_statistics.csv",
        index=False
    )

    # Dataset-level averages and spread across images
    weight_summary = (
        combined.groupby("dataset")
        .agg(
            image_count=("weight_mean", "size"),
            mean_weight=("weight_mean", "mean"),
            mean_spatial_std=("weight_std", "mean"),
            std_of_image_means=("weight_mean", "std"),
            lowest_weight=("weight_min", "min"),
            highest_weight=("weight_max", "max")
        )
        .reindex(DATASETS)
    )
    weight_summary.to_csv(RESULTS_ROOT / "weight_summary.csv")
    print("\nWEIGHT STATISTICS BY DATASET")
    print(weight_summary.round(4).to_string())

    comparison = create_summary(combined)

    plot_comparison(comparison, "psnr")
    plot_comparison(comparison, "ssim")
    plot_comparison(comparison, "delta_e")

    correction_columns = [
        "mean_blue_correction", "max_blue_correction", "fraction_pixels_corrected",
        *[f"corrected_vs_adaptive_{m}" for m in ("psnr", "ssim", "delta_e")],
        *[f"correction_win_{m}" for m in ("psnr", "ssim", "delta_e")]
    ]
    summary = combined.groupby("dataset")[correction_columns].mean().reindex(DATASETS)
    summary.to_csv(RESULTS_ROOT / "correction_summary.csv")
    print("\nCORRECTION SUMMARY (positive gains are improvements)")
    print(summary.round(5).to_string())
    for metric in ("psnr", "ssim", "delta_e"):
        ax = summary[f"corrected_vs_adaptive_{metric}"].plot.bar(rot=0)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(f"Mean {metric} improvement")
        ax.set_title("Corrected vs uncorrected adaptive")
        plt.tight_layout()
        plt.savefig(RESULTS_ROOT / f"correction_gain_{metric}.png", dpi=200)
        plt.close()

    print("\nEvaluation complete.")
    print(f"Results saved to: {RESULTS_ROOT.resolve()}")


if __name__ == "__main__":
    main()
