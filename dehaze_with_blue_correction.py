
"""DCP baseline, experimental local-feature blending,
and blue tint correction (RGB, float32 [0, 1]).
"""

from pathlib import Path
import argparse
import json

import cv2
import numpy as np


def correct_blue_tint(
    original,
    dehazed,
    brightness_threshold=0.7,
    saturation_threshold=0.2,
    blue_threshold=0.02,
    strength=0.75,
    radius=3
):
    """Reduce excessive blue in originally bright, neutral regions."""

    if not 0 <= brightness_threshold < 1:
        raise ValueError(
            "brightness_threshold must be in [0, 1)"
        )

    if not 0 < saturation_threshold <= 1:
        raise ValueError(
            "saturation_threshold must be in (0, 1]"
        )

    if not 0 <= blue_threshold <= 1:
        raise ValueError(
            "blue_threshold must be in [0, 1]"
        )

    if not 0 <= strength <= 1:
        raise ValueError(
            "strength must be in [0, 1]"
        )

    if radius < 0:
        raise ValueError(
            "Correction radius must be non-negative"
        )

    hsv = cv2.cvtColor(
        original.astype(np.float32),
        cv2.COLOR_RGB2HSV
    )

    brightness = hsv[..., 2]
    saturation = hsv[..., 1]

    # Confidence that the original pixel is bright.
    brightness_weight = np.clip(
        (brightness - brightness_threshold)
        / (1.0 - brightness_threshold),
        0,
        1
    )

    # Confidence that the original pixel is neutral.
    saturation_weight = np.clip(
        1.0 - saturation / saturation_threshold,
        0,
        1
    )

    # Combined confidence that the original pixel is white.
    # white_weight = brightness_weight * saturation_weight
    white_weight = np.sqrt(
        brightness_weight * saturation_weight
    )

    # Smooth the weight map to avoid abrupt boundaries.
    if radius > 0:
        white_weight = cv2.GaussianBlur(
            white_weight,
            (2 * radius + 1, 2 * radius + 1),
            0
        )

    # Identify excessive blue in the dehazed image.
    red = dehazed[..., 0]
    green = dehazed[..., 1]
    blue = dehazed[..., 2]

    blue_excess = np.maximum(
        blue - (red + green) / 2.0 - blue_threshold,
        0
    )

    # Calculate the required correction.
    correction = strength * white_weight * blue_excess

    # Preserve the original dehazed image.
    corrected = dehazed.copy()

    # Reduce only the blue channel.
    corrected[..., 2] -= correction

    return np.clip(corrected, 0, 1), correction


def mean_filter(image, radius):
    """Compute a local mean using a box filter."""

    return cv2.boxFilter(
        image,
        -1,
        (2 * radius + 1,) * 2,
        borderType=cv2.BORDER_REFLECT
    )


def guided_filter(guide, source, radius=20, eps=1e-3):
    """Grayscale guided filter.

    Guide and source must use the same [0, 1] scale.
    """

    mg = mean_filter(guide, radius)
    mp = mean_filter(source, radius)

    variance = np.maximum(
        mean_filter(guide * guide, radius) - mg * mg,
        0
    )

    covariance = (
        mean_filter(guide * source, radius) - mg * mp
    )

    a = covariance / (variance + eps)
    b = mp - a * mg

    return (
        mean_filter(a, radius) * guide
        + mean_filter(b, radius)
    )


def dark_channel(image, patch=15):
    """Calculate the dark channel of an RGB image."""

    return cv2.erode(
        image.min(axis=2),
        np.ones((patch, patch), np.uint8),
        borderType=cv2.BORDER_REFLECT
    )


def dcp(
    image,
    patch=15,
    omega=0.95,
    t_min=0.1,
    radius=20,
    eps=1e-3
):
    """Perform standard Dark Channel Prior dehazing."""

    dark = dark_channel(image, patch)

    # Select the top 0.1% of dark-channel pixels.
    count = max(1, int(dark.size * 0.001))

    candidates = np.argpartition(
        dark.ravel(),
        -count
    )[-count:]

    pixels = image.reshape(-1, 3)

    # Estimate atmospheric light.
    airlight = pixels[
        candidates[
            np.argmax(
                pixels[candidates].sum(axis=1)
            )
        ]
    ]

    airlight = np.maximum(airlight, 1e-3)

    # Estimate raw transmission.
    raw = 1 - omega * dark_channel(
        image / airlight,
        patch
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Refine the transmission map.
    transmission = np.clip(
        guided_filter(gray, raw, radius, eps),
        t_min,
        1
    )

    # Recover the scene radiance.
    restored = (
        (image - airlight) / transmission[..., None]
        + airlight
    )

    return (
        np.clip(restored, 0, 1),
        transmission,
        airlight
    )


def local_weights(
    image,
    patch=15,
    radius=20,
    eps=1e-3,
    brightness_coef=1.5,
    saturation_coef=-1.0,
    contrast_coef=-1.0,
    gradient_coef=-0.5,
    bias=-0.5,
    k=1.0
):
    """Calculate adaptive blending weights using local features."""

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    r = patch // 2

    # Local brightness.
    brightness = mean_filter(
        hsv[..., 2],
        r
    )

    # Local saturation.
    saturation = mean_filter(
        hsv[..., 1],
        r
    )

    # Local contrast.
    contrast = np.sqrt(
        np.maximum(
            mean_filter(gray * gray, r)
            - mean_filter(gray, r) ** 2,
            0
        )
    )

    # Local gradient magnitude.
    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
        scale=1 / 8
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
        scale=1 / 8
    )

    gradient = mean_filter(
        cv2.magnitude(gx, gy),
        r
    )

    # Fixed normalization scales.
    c = np.clip(contrast / 0.25, 0, 1)
    g = np.clip(gradient / 0.25, 0, 1)

    # Combined local feature score.
    score = (
        bias
        + brightness_coef * brightness
        + saturation_coef * saturation
        + contrast_coef * c
        + gradient_coef * g
    )

    # Convert scores into blending weights.
    raw = 1 / (
        1 + np.exp(
            -np.clip(k * score, -60, 60)
        )
    )

    # Refine weights using the guided filter.
    refined = np.clip(
        guided_filter(gray, raw, radius, eps),
        0,
        1
    )

    features = dict(
        brightness=brightness,
        saturation=saturation,
        contrast=c,
        gradient=g
    )

    return refined, raw, features


def process(
    image,
    patch=15,
    omega=0.95,
    t_min=0.1,
    radius=20,
    eps=1e-3,
    alpha=0.5,
    brightness_coef=1.5,
    saturation_coef=-1.0,
    contrast_coef=-1.0,
    gradient_coef=-0.5,
    bias=-0.5,
    k=1.0,
    brightness_threshold=0.7,
    saturation_threshold=0.2,
    blue_threshold=0.02,
    correction_strength=0.75,
    correction_radius=3
):
    """Run DCP, adaptive blending and blue tint correction."""

    # Validate input image.
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.size == 0
    ):
        raise ValueError(
            "Expected a nonempty H x W x 3 RGB image"
        )

    if (
        not np.isfinite(image).all()
        or image.min() < 0
        or image.max() > 1
    ):
        raise ValueError(
            "Input must be finite and scaled to [0, 1]"
        )

    # Validate DCP and filtering parameters.
    if (
        patch < 1
        or patch % 2 == 0
        or radius < 0
        or eps <= 0
    ):
        raise ValueError(
            "patch must be positive and odd; "
            "radius >= 0; eps > 0"
        )

    if (
        not 0 <= omega <= 1
        or not 0 < t_min <= 1
        or not 0 <= alpha <= 1
    ):
        raise ValueError(
            "Require omega/alpha in [0,1] "
            "and t_min in (0,1]"
        )

    if not np.isfinite([
        brightness_coef,
        saturation_coef,
        contrast_coef,
        gradient_coef,
        bias,
        k
    ]).all():
        raise ValueError(
            "Feature coefficients, bias and k must be finite"
        )

    if k < 0:
        raise ValueError(
            "Sigmoid strength k must be non-negative"
        )

    image = image.astype(np.float32)

    # -----------------------------------------------------
    # 1. Standard Dark Channel Prior.
    # -----------------------------------------------------

    baseline, transmission, airlight = dcp(
        image,
        patch,
        omega,
        t_min,
        radius,
        eps
    )

    # -----------------------------------------------------
    # 2. Calculate adaptive blending weights.
    # -----------------------------------------------------

    weight, raw, features = local_weights(
        image,
        patch,
        radius,
        eps,
        brightness_coef=brightness_coef,
        saturation_coef=saturation_coef,
        contrast_coef=contrast_coef,
        gradient_coef=gradient_coef,
        bias=bias,
        k=k
    )

    # -----------------------------------------------------
    # 3. Fixed blending.
    # -----------------------------------------------------

    fixed = (
        (1 - alpha) * image
        + alpha * baseline
    )

    # -----------------------------------------------------
    # 4. Mean-matched fixed blending.
    # -----------------------------------------------------

    mean_weight = float(np.mean(weight))

    matched_fixed = (
        (1 - mean_weight) * image
        + mean_weight * baseline
    )

    # -----------------------------------------------------
    # 5. Adaptive blending.
    # -----------------------------------------------------

    adaptive = (
        (1 - weight[..., None]) * image
        + weight[..., None] * baseline
    )

    # -----------------------------------------------------
    # 6. Blue tint correction.
    # -----------------------------------------------------

    adaptive_corrected, correction_map = correct_blue_tint(
        image,
        adaptive,
        brightness_threshold=brightness_threshold,
        saturation_threshold=saturation_threshold,
        blue_threshold=blue_threshold,
        strength=correction_strength,
        radius=correction_radius
    )

    # -----------------------------------------------------
    # 7. Return all outputs and intermediate maps.
    # -----------------------------------------------------

    return dict(
        original=image,
        dcp=baseline,
        fixed=fixed,
        matched_fixed=matched_fixed,
        adaptive=adaptive,
        adaptive_corrected=adaptive_corrected,
        correction_map=correction_map,
        weights=weight,
        raw_weights=raw,
        transmission=transmission,
        airlight=airlight,
        **features
    )


def save_rgb(path, image):
    """Save an RGB image or grayscale map."""

    data = np.round(
        np.clip(image, 0, 1) * 255
    ).astype(np.uint8)

    if data.ndim == 3:
        data = cv2.cvtColor(
            data,
            cv2.COLOR_RGB2BGR
        )

    if not cv2.imwrite(str(path), data):
        raise OSError(
            f"Could not write {path}"
        )


def save_results(results, output):
    """Save individual results and a comparison board."""

    output = Path(output)

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save all individual images and maps.
    for name, data in results.items():
        if name != "airlight":
            save_rgb(
                output / f"{name}.png",
                data
            )

    # Save an additional normalized correction map
    # for easier visual inspection.
    correction = results["correction_map"]

    max_correction = float(correction.max())

    if max_correction > 0:
        normalized_correction = (
            correction / max_correction
        )
    else:
        normalized_correction = np.zeros_like(
            correction
        )

    save_rgb(
        output / "correction_map_normalized.png",
        normalized_correction
    )

    # Create comparison tiles.
    tiles = []

    for name in (
        "original",
        "dcp",
        "fixed",
        "matched_fixed",
        "adaptive",
        "adaptive_corrected",
        "weights",
        "correction_map"
    ):
        data = results[name]

        # Use normalized correction map on the comparison
        # board so small corrections are visible.
        if name == "correction_map":
            data = normalized_correction
            label = "correction_map (normalized)"
        else:
            label = name

        # Convert grayscale maps to three channels.
        if data.ndim == 2:
            data = np.repeat(
                data[..., None],
                3,
                axis=2
            )

        h, w = data.shape[:2]

        tile = np.zeros(
            (h + 32, w, 3),
            np.uint8
        )

        tile[32:] = np.round(
            np.clip(data, 0, 1) * 255
        ).astype(np.uint8)

        cv2.putText(
            tile,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        tiles.append(tile)

    # Two rows of four images.
    board = np.vstack((
        np.hstack(tiles[:4]),
        np.hstack(tiles[4:])
    ))

    save_rgb(
        output / "comparison.png",
        board.astype(np.float32) / 255
    )


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # Input and output.
    parser.add_argument(
        "input",
        type=Path
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results")
    )

    # DCP and filtering parameters.
    parser.add_argument(
        "--patch",
        type=int,
        default=15
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=20
    )

    parser.add_argument(
        "--omega",
        type=float,
        default=0.95
    )

    parser.add_argument(
        "--t-min",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5
    )

    # Adaptive blending parameters.
    parser.add_argument(
        "--brightness-coef",
        type=float,
        default=1.5
    )

    parser.add_argument(
        "--saturation-coef",
        type=float,
        default=-1.0
    )

    parser.add_argument(
        "--contrast-coef",
        type=float,
        default=-1.0
    )

    parser.add_argument(
        "--gradient-coef",
        type=float,
        default=-0.5
    )

    parser.add_argument(
        "--bias",
        type=float,
        default=-0.5
    )

    parser.add_argument(
        "--k",
        type=float,
        default=1.0
    )

    # Blue tint correction parameters.
    parser.add_argument(
        "--brightness-threshold",
        type=float,
        default=0.7
    )

    parser.add_argument(
        "--saturation-threshold",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--blue-threshold",
        type=float,
        default=0.02
    )

    parser.add_argument(
        "--correction-strength",
        type=float,
        default=0.75
    )

    parser.add_argument(
        "--correction-radius",
        type=int,
        default=3
    )

    args = parser.parse_args()

    # Load input image.
    raw = cv2.imread(
        str(args.input),
        cv2.IMREAD_COLOR
    )

    if raw is None:
        parser.error(
            f"Cannot read image: {args.input}"
        )

    # Convert BGR uint8 to RGB float32.
    image = cv2.cvtColor(
        raw,
        cv2.COLOR_BGR2RGB
    ).astype(np.float32) / 255

    # Run dehazing.
    try:
        results = process(
            image,
            patch=args.patch,
            omega=args.omega,
            t_min=args.t_min,
            radius=args.radius,
            eps=args.eps,
            alpha=args.alpha,
            brightness_coef=args.brightness_coef,
            saturation_coef=args.saturation_coef,
            contrast_coef=args.contrast_coef,
            gradient_coef=args.gradient_coef,
            bias=args.bias,
            k=args.k,
            brightness_threshold=args.brightness_threshold,
            saturation_threshold=args.saturation_threshold,
            blue_threshold=args.blue_threshold,
            correction_strength=args.correction_strength,
            correction_radius=args.correction_radius
        )

    except ValueError as exc:
        parser.error(str(exc))

    # Save generated images.
    save_results(
        results,
        args.output
    )

    # Calculate correction statistics.
    correction = results["correction_map"]

    mean_blue_correction = float(
        correction.mean()
    )

    max_blue_correction = float(
        correction.max()
    )

    fraction_pixels_corrected = float(
        (correction > 1e-6).mean()
    )

    # Save experiment configuration and statistics.
    metadata = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    metadata.update(
        airlight=results["airlight"].tolist(),
        mean_weight=float(results["weights"].mean()),
        spatial_weight_std=float(results["weights"].std()),
        mean_blue_correction=mean_blue_correction,
        max_blue_correction=max_blue_correction,
        fraction_pixels_corrected=fraction_pixels_corrected,
        status="Experimental dehazing and colour correction."
    )

    (args.output / "run.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8"
    )

    print(
        f"Saved comparison and intermediate maps "
        f"to {args.output}"
    )

    print("\nBlue correction statistics:")

    print(
        f"Mean blue correction: "
        f"{mean_blue_correction:.6f}"
    )

    print(
        f"Maximum blue correction: "
        f"{max_blue_correction:.6f}"
    )

    print(
        f"Percentage of pixels corrected: "
        f"{100 * fraction_pixels_corrected:.2f}%"
    )


if __name__ == "__main__":
    main()
