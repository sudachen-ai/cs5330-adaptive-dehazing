
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = RESULTS_ROOT / "comparisons"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    "baseline",
    "no_bias",
    "no_brightness",
    "no_saturation",
    "no_contrast",
    "no_gradient"
]

DATASETS = [
    "NH-HAZE",
    "O-HAZE"
]

METHODS = [
    "original",
    "dcp",
    "fixed",
    "matched_fixed",
    "adaptive"
]

METRICS = [
    "psnr",
    "ssim"
]

EXPERIMENT_LABELS = {
    "baseline": "Baseline",
    "no_bias": "No Bias",
    "no_brightness": "No Brightness",
    "no_saturation": "No Saturation",
    "no_contrast": "No Contrast",
    "no_gradient": "No Gradient"
}


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():

    records = []

    for experiment in EXPERIMENTS:

        csv_path = (
            RESULTS_ROOT
            / experiment
            / "method_comparison.csv"
        )

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing results file: {csv_path}"
            )

        df = pd.read_csv(csv_path)

        required_columns = {
            "dataset",
            "method",
            "mean_psnr",
            "mean_ssim"
        }

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{csv_path} is missing columns: {missing}"
            )

        df["experiment"] = experiment

        records.append(df)

    combined = pd.concat(
        records,
        ignore_index=True
    )

    combined.to_csv(
        OUTPUT_DIR / "all_ablation_results.csv",
        index=False
    )

    return combined


# ============================================================
# PLOT ONE METHOD AND METRIC
# ============================================================

def plot_method(combined, method, metric):

    method_df = combined[
        combined["method"] == method
    ]

    pivot = method_df.pivot(
        index="experiment",
        columns="dataset",
        values=f"mean_{metric}"
    )

    pivot = pivot.reindex(
        index=EXPERIMENTS,
        columns=DATASETS
    )

    if pivot.isna().any().any():
        raise ValueError(
            f"Missing data for {method}, {metric}"
        )

    x = np.arange(len(EXPERIMENTS))

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    for i, dataset in enumerate(DATASETS):

        positions = (
            x
            + (i - (len(DATASETS) - 1) / 2) * width
        )

        ax.bar(
            positions,
            pivot[dataset].values,
            width,
            label=dataset
        )

    ax.set_title(
        f"{method.replace('_', ' ').title()} "
        f"- Feature Ablation ({metric.upper()})"
    )

    ax.set_xlabel("Experiment")

    if metric == "psnr":
        ax.set_ylabel("Mean PSNR (dB)")
    else:
        ax.set_ylabel("Mean SSIM")

    ax.set_xticks(x)

    ax.set_xticklabels(
        [EXPERIMENT_LABELS[e] for e in EXPERIMENTS],
        rotation=20,
        ha="right"
    )

    ax.legend(title="Dataset")

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{method}_{metric}_ablation.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    print(f"Saved: {output_path}")


# ============================================================
# CREATE SUMMARY OF CHANGES FROM BASELINE
# ============================================================

def create_delta_summary(combined):

    baseline = (
        combined[
            combined["experiment"] == "baseline"
        ]
        .set_index(["dataset", "method"])
    )

    records = []

    for _, row in combined.iterrows():

        dataset = row["dataset"]
        method = row["method"]
        experiment = row["experiment"]

        baseline_row = baseline.loc[
            (dataset, method)
        ]

        records.append({
            "experiment": experiment,
            "dataset": dataset,
            "method": method,
            "psnr_change": (
                row["mean_psnr"]
                - baseline_row["mean_psnr"]
            ),
            "ssim_change": (
                row["mean_ssim"]
                - baseline_row["mean_ssim"]
            )
        })

    delta_df = pd.DataFrame(records)

    delta_df.to_csv(
        OUTPUT_DIR / "changes_from_baseline.csv",
        index=False
    )

    return delta_df


# ============================================================
# MAIN
# ============================================================

def main():

    combined = load_results()

    for method in METHODS:
        for metric in METRICS:

            plot_method(
                combined,
                method,
                metric
            )

    delta_df = create_delta_summary(combined)

    print("\n" + "=" * 70)
    print("ADAPTIVE METHOD - CHANGES FROM BASELINE")
    print("=" * 70)

    adaptive_deltas = delta_df[
        delta_df["method"] == "adaptive"
    ]

    print(
        adaptive_deltas.round(4).to_string(
            index=False
        )
    )

    print("\nComparison complete.")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
