r"""Why does adaptive blending lose to matched_fixed on RESIDE?

Run from the repository root:
    python analyze_reside_failure.py [--brightness 3.0] [--out analysis_reside]

For each image it splits the MSE gap between adaptive and matched_fixed into
two exact terms. With weight w = m + d (m = mean weight, d = deviation),
D = dcp - original and r = matched_fixed - gt:

    MSE(adaptive) - MSE(matched) = 2 * mean(d * r * D)  +  mean(d^2 * D^2)
                                   \_ alignment _/        \_ variance penalty _/

The variance penalty is never negative: varying the weight only pays off if the
alignment term is negative enough, i.e. if the weight is high exactly where
DCP is locally better. The script also computes a locally optimal ("oracle")
weight map from the ground truth, to show how much spatial adaptivity could
gain at best, and how well the heuristic weight map agrees with it.
"""
from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd

from dehaze import process, mean_filter

ROOT = Path("datasets/datasets_quarter_size")
SETS = [("NH-HAZE", "outdoor", "nhhaze"), ("O-HAZE", "outdoor", "ohaze"),
        ("RESIDE", "indoor", "reside"), ("RESIDE", "outdoor", "reside")]
COMMON = dict(patch=15, omega=0.90, t_min=0.1, radius=20, eps=1e-3, alpha=0.5,
              saturation_coef=-1.0, contrast_coef=-1.0, gradient_coef=-0.5,
              bias=-0.5, k=1.0)


def load_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def psnr(mse):
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def oracle_weight(hazy, dcp, gt, radius=7):
    """Least-squares blend weight in each (2r+1)^2 window, using the GT."""
    D = (dcp - hazy).astype(np.float32)
    e = (gt - hazy).astype(np.float32)
    num = mean_filter((e * D).sum(axis=2), radius)
    den = mean_filter((D * D).sum(axis=2), radius)
    return np.clip(num / (den + 1e-6), 0, 1)


def analyze_image(hazy, gt, brightness_coef):
    res = process(hazy, brightness_coef=brightness_coef, **COMMON)
    w = res["weights"]
    m = float(w.mean())
    d = (w - m)[..., None]
    D = res["dcp"] - hazy
    r = res["matched_fixed"] - gt
    align = float(2 * np.mean(d * r * D))
    penalty = float(np.mean(d * d * D * D))
    mse_ad = float(np.mean((res["adaptive"] - gt) ** 2))
    mse_mf = float(np.mean((res["matched_fixed"] - gt) ** 2))

    w_star = oracle_weight(hazy, res["dcp"], gt)
    oracle = (1 - w_star[..., None]) * hazy + w_star[..., None] * res["dcp"]
    mse_or = float(np.mean((oracle - gt) ** 2))

    # Bright, flat regions (mostly sky or white walls): where the
    # brightness term pushes the weight up and DCP is known to struggle.
    bright_flat = (res["brightness"] > 0.7) & (res["gradient"] < 0.08)
    frac = float(bright_flat.mean())
    if bright_flat.any():
        w_bf = float(w[bright_flat].mean())
        w_star_bf = float(w_star[bright_flat].mean())
        w_rest = float(w[~bright_flat].mean()) if (~bright_flat).any() else np.nan
        w_star_rest = float(w_star[~bright_flat].mean()) if (~bright_flat).any() else np.nan
    else:
        w_bf = w_star_bf = w_rest = w_star_rest = np.nan

    return dict(
        weight_mean=m, weight_std=float(w.std()),
        psnr_matched=psnr(mse_mf), psnr_adaptive=psnr(mse_ad),
        psnr_dcp=psnr(float(np.mean((res["dcp"] - gt) ** 2))),
        psnr_oracle=psnr(mse_or),
        delta_mse=mse_ad - mse_mf, align=align, penalty=penalty,
        decomp_error=(mse_ad - mse_mf) - (align + penalty),
        corr_w_oracle=float(np.corrcoef(w.ravel(), w_star.ravel())[0, 1]),
        oracle_mean=float(w_star.mean()),
        bright_flat_frac=frac, w_bright_flat=w_bf, w_star_bright_flat=w_star_bf,
        w_rest=w_rest, w_star_rest=w_star_rest,
    ), res, w_star


def pairs(dataset, category, tag):
    hazy_dir = ROOT / dataset / "hazy" / category
    for hp in sorted(hazy_dir.glob("*.jpg")):
        gp = ROOT / dataset / "GT" / category / hp.name.replace("_hazy_", "_GT_", 1)
        if gp.exists():
            yield hp, gp


def to_u8(x):
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.round(np.clip(x, 0, 1) * 255).astype(np.uint8)


def panel(hazy, gt, res, w_star, title):
    """Row: hazy | GT | DCP | adaptive | matched | weight | oracle weight."""
    tiles = [("hazy", hazy), ("GT", gt), ("DCP", res["dcp"]),
             ("adaptive", res["adaptive"]), ("matched_fixed", res["matched_fixed"]),
             ("weight (ours)", res["weights"]), ("oracle weight", w_star)]
    out = []
    for name, img in tiles:
        t = to_u8(img)
        bar = np.zeros((26, t.shape[1], 3), np.uint8)
        cv2.putText(bar, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        out.append(np.vstack([bar, t]))
    row = np.hstack(out)
    head = np.zeros((26, row.shape[1], 3), np.uint8)
    cv2.putText(head, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 0), 1, cv2.LINE_AA)
    return np.vstack([head, row])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brightness", type=float, default=3.0)
    ap.add_argument("--out", type=Path, default=Path("analysis_reside"))
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset, category, tag in SETS:
        for hp, gp in pairs(dataset, category, tag):
            row, _, _ = analyze_image(load_rgb(hp), load_rgb(gp), args.brightness)
            rows.append(dict(dataset=dataset, category=category, image=hp.name, **row))
        print("done", dataset, category, flush=True)
    frame = pd.DataFrame(rows)
    frame["gain_adaptive"] = frame.psnr_adaptive - frame.psnr_matched
    frame["gain_oracle"] = frame.psnr_oracle - frame.psnr_matched
    frame.to_csv(args.out / "per_image.csv", index=False)

    agg = frame.groupby(["dataset", "category"]).agg(
        n=("image", "size"),
        adaptive_win_rate=("gain_adaptive", lambda s: (s > 0).mean()),
        gain_adaptive_db=("gain_adaptive", "mean"),
        gain_oracle_db=("gain_oracle", "mean"),
        align=("align", "mean"), penalty=("penalty", "mean"),
        corr_w_oracle=("corr_w_oracle", "mean"),
        weight_mean=("weight_mean", "mean"), oracle_mean=("oracle_mean", "mean"),
        bright_flat_frac=("bright_flat_frac", "mean"),
        w_bright_flat=("w_bright_flat", "mean"),
        w_star_bright_flat=("w_star_bright_flat", "mean"),
        w_rest=("w_rest", "mean"), w_star_rest=("w_star_rest", "mean"),
        max_decomp_error=("decomp_error", lambda s: s.abs().max()),
    ).reset_index()
    agg.to_csv(args.out / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(agg.round(4).to_string(index=False))

    # Example panels: worst losses on RESIDE outdoor, plus the best case.
    out_rows = frame[(frame.dataset == "RESIDE") & (frame.category == "outdoor")]
    picks = list(out_rows.nsmallest(args.examples, "gain_adaptive").image)
    picks += list(out_rows.nlargest(1, "gain_adaptive").image)
    boards = []
    for name in picks:
        hp = ROOT / "RESIDE" / "hazy" / "outdoor" / name
        gp = ROOT / "RESIDE" / "GT" / "outdoor" / name.replace("_hazy_", "_GT_", 1)
        hazy, gt = load_rgb(hp), load_rgb(gp)
        row, res, w_star = analyze_image(hazy, gt, args.brightness)
        title = (f"{name}  adaptive-matched {row['psnr_adaptive'] - row['psnr_matched']:+.2f} dB"
                 f"  oracle-matched {row['psnr_oracle'] - row['psnr_matched']:+.2f} dB"
                 f"  corr(w, oracle) {row['corr_w_oracle']:+.2f}")
        board = panel(hazy, gt, res, w_star, title)
        boards.append(board)
    width = max(b.shape[1] for b in boards)
    boards = [np.pad(b, ((0, 4), (0, width - b.shape[1]), (0, 0))) for b in boards]
    cv2.imwrite(str(args.out / "reside_outdoor_examples.png"),
                cv2.cvtColor(np.vstack(boards), cv2.COLOR_RGB2BGR))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
