"""DCP baseline and experimental local-feature blending (RGB, float32 [0, 1])."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np


def mean_filter(image, radius):
    return cv2.boxFilter(image, -1, (2 * radius + 1,) * 2,
                         borderType=cv2.BORDER_REFLECT)


def guided_filter(guide, source, radius=20, eps=1e-3):
    """Grayscale guided filter; guide and source must use the same [0,1] scale."""
    mg, mp = mean_filter(guide, radius), mean_filter(source, radius)
    variance = np.maximum(mean_filter(guide * guide, radius) - mg * mg, 0)
    covariance = mean_filter(guide * source, radius) - mg * mp
    a = covariance / (variance + eps)
    b = mp - a * mg
    return mean_filter(a, radius) * guide + mean_filter(b, radius)


def dark_channel(image, patch=15):
    return cv2.erode(image.min(axis=2), np.ones((patch, patch), np.uint8),
                     borderType=cv2.BORDER_REFLECT)


def dcp(image, patch=15, omega=0.95, t_min=0.1, radius=20, eps=1e-3):
    dark = dark_channel(image, patch)
    count = max(1, int(dark.size * 0.001))
    candidates = np.argpartition(dark.ravel(), -count)[-count:]
    pixels = image.reshape(-1, 3)
    airlight = pixels[candidates[np.argmax(pixels[candidates].sum(axis=1))]]
    airlight = np.maximum(airlight, 1e-3)
    raw = 1 - omega * dark_channel(image / airlight, patch)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    transmission = np.clip(guided_filter(gray, raw, radius, eps), t_min, 1)
    restored = (image - airlight) / transmission[..., None] + airlight
    return np.clip(restored, 0, 1), transmission, airlight


def local_weights(image, patch=15, radius=20, eps=1e-3, brightness_coef=1.5, saturation_coef=-1.0,
                  contrast_coef=-1.0, gradient_coef=-0.5, bias=-0.5):
    """Dense overlapping windows; coefficients are initial, unvalidated defaults."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    r = patch // 2
    brightness = mean_filter(hsv[..., 2], r)
    saturation = mean_filter(hsv[..., 1], r)
    contrast = np.sqrt(np.maximum(mean_filter(gray * gray, r)
                                  - mean_filter(gray, r) ** 2, 0))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    gradient = mean_filter(cv2.magnitude(gx, gy), r)
    # Fixed scales avoid fitting normalization on test images.
    c = np.clip(contrast / 0.25, 0, 1)
    g = np.clip(gradient / 0.25, 0, 1)
    score = (bias + brightness_coef * brightness + saturation_coef * saturation + contrast_coef * c
             + gradient_coef * g)
    raw = 1 / (1 + np.exp(-score))
    refined = np.clip(guided_filter(gray, raw, radius, eps), 0, 1)
    features = dict(brightness=brightness, saturation=saturation,
                    contrast=c, gradient=g)
    return refined, raw, features


def process(image, patch=15, omega=0.95, t_min=0.1, radius=20,
            eps=1e-3, alpha=0.5, brightness_coef=1.5, saturation_coef=-1.0, contrast_coef=-1.0,
            gradient_coef=-0.5, bias=-0.5):
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("Expected a nonempty H x W x 3 RGB image")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Input must be finite and scaled to [0, 1]")
    if patch < 1 or patch % 2 == 0 or radius < 0 or eps <= 0:
        raise ValueError("patch must be positive and odd; radius >= 0; eps > 0")
    if not 0 <= omega <= 1 or not 0 < t_min <= 1 or not 0 <= alpha <= 1:
        raise ValueError("Require omega/alpha in [0,1] and t_min in (0,1]")
    image = image.astype(np.float32)
    baseline, transmission, airlight = dcp(image, patch, omega, t_min, radius, eps)
    weight, raw, features = local_weights(image, patch, radius, eps, brightness_coef=brightness_coef,
        saturation_coef=saturation_coef, contrast_coef=contrast_coef, gradient_coef=gradient_coef, bias=bias)
    adaptive = (1 - weight[..., None]) * image + weight[..., None] * baseline
    fixed = (1 - alpha) * image + alpha * baseline
    mean_weight = float(np.mean(weight))
    matched_fixed = ((1 - mean_weight) * image+ mean_weight * baseline)
    return dict(original=image, dcp=baseline, fixed=fixed, matched_fixed=matched_fixed,
                adaptive=adaptive, weights=weight, raw_weights=raw, transmission=transmission,
                airlight=airlight, **features)


def save_rgb(path, image):
    data = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    if data.ndim == 3:
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), data):
        raise OSError(f"Could not write {path}")


def save_results(results, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in results.items():
        if name != "airlight":
            save_rgb(output / f"{name}.png", data)
    tiles = []
    for name in ("original", "dcp", "fixed", "adaptive", "weights", "transmission"):
        data = results[name]
        if data.ndim == 2:
            data = np.repeat(data[..., None], 3, axis=2)
        h, w = data.shape[:2]
        tile = np.zeros((h + 32, w, 3), np.uint8)
        tile[32:] = np.round(data * 255).astype(np.uint8)
        cv2.putText(tile, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    board = np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:])))
    save_rgb(output / "comparison.png", board.astype(np.float32) / 255)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--patch", type=int, default=15)
    parser.add_argument("--radius", type=int, default=20)
    parser.add_argument("--omega", type=float, default=0.95)
    parser.add_argument("--t-min", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    raw = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if raw is None:
        parser.error(f"Cannot read image: {args.input}")
    image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    try:
        results = process(image, args.patch, args.omega, args.t_min,
                          args.radius, args.eps, args.alpha)
    except ValueError as exc:
        parser.error(str(exc))
    save_results(results, args.output)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(airlight=results["airlight"].tolist(),
                    mean_weight=float(results["weights"].mean()),
                    status="Prototype: heuristic coefficients have not been validated.")
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved comparison and intermediate maps to {args.output}")


if __name__ == "__main__":
    main()
