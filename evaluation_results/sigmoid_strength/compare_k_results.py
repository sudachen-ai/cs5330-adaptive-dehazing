"""Compare sigmoid-strength experiments.

Save this file as:
    evaluation_results/sigmoid_strength/compare_k_results.py

Expected sibling folders: k_0_5, k_1, k_2, k_4, k_8.
Run from any working directory: python compare_k_results.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'comparison'
OUTPUT.mkdir(parents=True, exist_ok=True)
EXPERIMENTS = {0.5: 'k_0_5', 1.0: 'k_1', 2.0: 'k_2', 4.0: 'k_4', 8.0: 'k_8'}
METHODS = {}
DATASETS = ['NH-HAZE', 'O-HAZE']
METHODS = ['original', 'dcp', 'fixed', 'matched_fixed', 'adaptive']


def load_results():
    image_frames = []
    method_frames = []
    weight_frames = []
    for k, folder in EXPERIMENTS.items():
        directory = ROOT / folder
        image_file = directory / 'all_image_metrics.csv'
        method_file = directory / 'method_comparison.csv'
        weight_file = directory / 'weight_summary.csv'
        for path in (image_file, method_file, weight_file):
            if not path.is_file():
                raise FileNotFoundError(f'Missing {path}\nRun the evaluation for k={k} first.')
        images = pd.read_csv(image_file)
        methods = pd.read_csv(method_file)
        weights = pd.read_csv(weight_file)
        for df in (images, methods, weights):
            df['k'] = k
        image_frames.append(images)
        method_frames.append(methods)
        weight_frames.append(weights)
    return (pd.concat(image_frames, ignore_index=True),
            pd.concat(method_frames, ignore_index=True),
            pd.concat(weight_frames, ignore_index=True))


def plot_lines(df, value, title, filename, *, series='method', ylabel=None, zero_line=False):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if series == 'method':
        for dataset in DATASETS:
            for method in METHODS:
                part = df[(df.dataset == dataset) & (df.method == method)].sort_values('k')
                ax.plot(part.k, part[value], marker='o', label=f'{dataset}: {method}')
    else:
        for dataset in DATASETS:
            part = df[df.dataset == dataset].sort_values('k')
            ax.plot(part.k, part[value], marker='o', label=dataset)
    if zero_line:
        ax.axhline(0, color='gray', linewidth=1, linestyle='--')
    ax.set_xticks(list(EXPERIMENTS))
    ax.set_xlabel('Sigmoid strength k')
    ax.set_ylabel(ylabel or value)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=200)
    plt.close(fig)


def main():
    images, methods, weights = load_results()
    images.to_csv(OUTPUT / 'all_k_image_metrics.csv', index=False)
    methods.to_csv(OUTPUT / 'all_k_method_comparison.csv', index=False)
    weights.to_csv(OUTPUT / 'all_k_weight_summary.csv', index=False)

    for metric in ('psnr', 'ssim'):
        plot_lines(methods, f'mean_{metric}', f'All methods: {metric.upper()} by k',
                   f'{metric}_all_methods.png', ylabel=f'Mean {metric.upper()}')
        for method in METHODS:
            subset = methods[methods.method == method]
            plot_lines(subset, f'mean_{metric}', f'{method}: {metric.upper()} by k',
                       f'{metric}_{method}_by_k.png', series='dataset', ylabel=f'Mean {metric.upper()}')

    for column, title, filename in (
        ('mean_weight', 'Mean adaptive weight by k', 'mean_weight_by_k.png'),
        ('mean_spatial_std', 'Mean spatial weight standard deviation by k', 'mean_spatial_std_by_k.png'),
        ('std_of_image_means', 'Between-image variation of mean weights by k', 'std_of_image_means_by_k.png'),
    ):
        plot_lines(weights, column, title, filename, series='dataset')

    summary = (images.groupby(['k', 'dataset'], as_index=False)
               .agg(image_count=('image', 'size'),
                    adaptive_vs_matched_psnr=('adaptive_vs_matched_psnr', 'mean'),
                    adaptive_vs_matched_ssim=('adaptive_vs_matched_ssim', 'mean'),
                    adaptive_psnr_win_fraction=('adaptive_vs_matched_psnr', lambda x: float((x > 0).mean())),
                    adaptive_ssim_win_fraction=('adaptive_vs_matched_ssim', lambda x: float((x > 0).mean()))))
    summary.to_csv(OUTPUT / 'adaptive_vs_matched_by_k.csv', index=False)
    for metric in ('psnr', 'ssim'):
        column = f'adaptive_vs_matched_{metric}'
        plot_lines(summary, column, f'Adaptive minus mean-matched fixed: {metric.upper()}',
                   f'{metric}_adaptive_minus_matched.png', series='dataset',
                   ylabel=f'Mean difference in {metric.upper()}', zero_line=True)

    baseline = methods[methods.k == 1.0][['dataset', 'method', 'mean_psnr', 'mean_ssim']]
    comparison = methods.merge(baseline, on=['dataset', 'method'], suffixes=('', '_k1'))
    comparison['psnr_change_vs_k1'] = comparison.mean_psnr - comparison.mean_psnr_k1
    comparison['ssim_change_vs_k1'] = comparison.mean_ssim - comparison.mean_ssim_k1
    comparison.to_csv(OUTPUT / 'changes_vs_k_1.csv', index=False)

    print('Comparison complete. Files saved to:', OUTPUT)
    print('\nAdaptive versus matched fixed:')
    print(summary.round(5).to_string(index=False))


if __name__ == '__main__':
    main()
