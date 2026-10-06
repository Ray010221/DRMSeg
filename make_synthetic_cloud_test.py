"""Build reproducible Optical-cloud stress-test splits for DRM-Seg.

The generated binary masks are evaluation ground truth only.  They are never
read by the model; candidate regions must be regenerated from the corrupted
optical images with SAM1/SAM2/Felzenszwalb.
"""

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm


SEVERITIES = {
    "thin": {"coverage": (0.08, 0.18), "opacity": (0.20, 0.36)},
    "medium": {"coverage": (0.16, 0.30), "opacity": (0.42, 0.62)},
    "thick": {"coverage": (0.24, 0.42), "opacity": (1, 1)},
    "opaque": {"coverage": (0.36, 0.60), "opacity": (1, 1)},
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ALIGNED_COPY_DIRS = ("sar", "lbl", "lblRGB", "lbl_RGB")


def parse_args():
    parser = argparse.ArgumentParser(description="Create a fixed synthetic Optical-cloud evaluation split.")
    parser.add_argument("--source-root", required=True, help="Existing split containing opt/sar/lbl.")
    parser.add_argument("--output-root", required=True, help="New split directory to create.")
    parser.add_argument("--severity", choices=tuple(SEVERITIES), required=True)
    parser.add_argument("--coverage", nargs=2, type=float, default=None, metavar=("MIN", "MAX"),
                        help="Override severity coverage range, e.g. --coverage 0.40 0.60.")
    parser.add_argument("--opacity", nargs=2, type=float, default=None, metavar=("MIN", "MAX"),
                        help="Override severity opacity range, e.g. --opacity 1.0 1.0.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--source-list", default=None,
                        help="Optional text file with one image stem per line; use visually verified clear Xi'an images.")
    parser.add_argument("--passthrough-list", default=None,
                        help="Optional stems copied unchanged into a mixed natural+synthetic split. "
                             "They are excluded from synthetic-mask metrics.")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--cloud-color", nargs=3, type=float, default=(246, 248, 250))
    parser.add_argument("--opaque-core-strength", type=float, default=0.58,
                        help="Dense-core alpha contribution for opaque clouds in [0, 1]; larger is brighter.")
    parser.add_argument("--backend", choices=("multiscale", "scg", "real-patch"), default="multiscale",
                        help="Cloud generator. multiscale is built in; scg calls the official package if installed.")
    parser.add_argument("--scg-path", default=None,
                        help="Optional local checkout of SatelliteCloudGenerator to add to sys.path.")
    parser.add_argument("--shadow", action="store_true",
                        help="Add a weak, displaced cloud shadow (disabled by default for mask-only experiments).")
    parser.add_argument("--cloud-source-image", default=None,
                        help="Optional real cloudy Optical image used as a cloud-texture source.")
    parser.add_argument("--cloud-source-mask", default=None,
                        help="Binary cloud support (.png or .npy) aligned with --cloud-source-image.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_stems(path):
    if path is None:
        return None
    return {
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _unit_interval(array):
    array = array.astype(np.float32)
    return (array - array.min()) / max(float(array.max() - array.min()), 1e-6)


def _fractal_noise(height, width, rng, octaves=(3, 5, 9, 17, 31)):
    """Multi-scale smooth noise with a different realization for every sample."""
    field = np.zeros((height, width), dtype=np.float32)
    weight_sum = 0.0
    for octave in octaves:
        grid_h = max(2, int(round(height / octave)))
        grid_w = max(2, int(round(width / octave)))
        grid = rng.normal(size=(grid_h, grid_w)).astype(np.float32)
        layer = cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)
        weight = 1.0 / np.sqrt(octave)
        field += weight * layer
        weight_sum += weight
    return _unit_interval(field / weight_sum)


def _locality_envelope(height, width, rng):
    """Generate one to three irregular local cloud systems instead of a full-frame veil."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    envelope = np.zeros((height, width), dtype=np.float32)
    for _ in range(int(rng.integers(1, 4))):
        # Allow centers just outside the image to obtain partial cloud incursions.
        center_x = rng.uniform(-0.12, 1.12) * width
        center_y = rng.uniform(-0.12, 1.12) * height
        radius_x = rng.uniform(0.14, 0.42) * width
        radius_y = rng.uniform(0.10, 0.36) * height
        angle = rng.uniform(0.0, np.pi)
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        x_shift = xx - center_x
        y_shift = yy - center_y
        x_rot = cos_angle * x_shift + sin_angle * y_shift
        y_rot = -sin_angle * x_shift + cos_angle * y_shift
        blob = np.exp(-0.5 * ((x_rot / radius_x) ** 2 + (y_rot / radius_y) ** 2))
        envelope = np.maximum(envelope, blob)
    return _unit_interval(envelope)


def multiscale_cloud_field(height, width, rng, coverage, opacity, opaque_core_strength):
    """Cloud layer inspired by satellite cloud simulation: locality + fractal detail.

    The binary map is derived from the exact alpha field and remains evaluation
    ground truth.  Unlike the earlier single thresholded noise map, this creates
    different multi-lobed systems at every seed.
    """
    coarse = _fractal_noise(height, width, rng, octaves=(10, 19, 37, 73))
    detail = _fractal_noise(height, width, rng, octaves=(3, 6, 12, 24))
    locality = _locality_envelope(height, width, rng)
    cloudness = locality * (0.68 * coarse + 0.32 * detail)
    cloudness = cv2.GaussianBlur(cloudness, (0, 0), sigmaX=max(0.8, min(height, width) / 180.0))
    threshold = np.quantile(cloudness, 1.0 - coverage)
    support = np.clip((cloudness - threshold) / max(cloudness.max() - threshold, 1e-6), 0.0, 1.0)
    # Preserve internal cloud texture but make the perimeter physically soft.
    support = cv2.GaussianBlur(support, (0, 0), sigmaX=max(1.2, min(height, width) / 115.0))
    support = _unit_interval(support) * (cloudness >= threshold * 0.80)
    if opacity >= 0.95:
        # Severe occlusion needs an opaque core.  The blurred binary support
        # keeps the outer transition natural while avoiding a thin haze.
        dense_core = cv2.GaussianBlur(
            (cloudness >= threshold).astype(np.float32),
            (0, 0),
            sigmaX=max(1.6, min(height, width) / 150.0),
        )
        dense_core = np.clip(dense_core, 0.0, 1.0)
        core_strength = float(np.clip(opaque_core_strength, 0.0, 1.0))
        alpha_shape = np.clip(
            core_strength * dense_core + (1.0 - core_strength) * support ** 0.30,
            0.0,
            1.0,
        )
    else:
        alpha_shape = 0.30 * support + 0.70 * support ** 0.72
    alpha = np.clip(opacity * alpha_shape, 0.0, 1.0)
    gt_mask = alpha >= max(0.075, opacity * 0.18)
    return alpha[..., None], gt_mask.astype(np.uint8)


def scg_cloud_field(optical, opacity, args):
    """Use the official SatelliteCloudGenerator when its source is locally available."""
    scg = None
    if args.scg_path:
        source_dir = Path(args.scg_path).resolve() / "src"
        init_file = source_dir / "__init__.py"
        if init_file.is_file():
            spec = importlib.util.spec_from_file_location(
                "satellite_cloud_generator", init_file, submodule_search_locations=[str(source_dir)]
            )
            scg = importlib.util.module_from_spec(spec)
            sys.modules["satellite_cloud_generator"] = scg
            spec.loader.exec_module(scg)
    try:
        if scg is None:
            import satellite_cloud_generator as scg
    except ImportError as error:
        raise RuntimeError(
            "The official SatelliteCloudGenerator is unavailable. Download its repository "
            "into third_party/SatelliteCloudGenerator and pass --scg-path, or use --backend multiscale."
        ) from error

    import torch
    source = torch.from_numpy(optical.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    cloudy, transparency = scg.add_cloud(
        source,
        min_lvl=max(0.0, opacity - 0.12),
        max_lvl=min(1.0, opacity + 0.12),
        clear_threshold=0.04,
        locality_degree=2,
        blur_scaling=1.0,
        cloud_color=True,
        return_cloud=True,
    )
    cloudy = cloudy.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0
    alpha = transparency.squeeze(0).float().mean(dim=0).detach().cpu().numpy()
    return np.clip(cloudy, 0.0, 255.0), alpha[..., None], (alpha >= 0.075).astype(np.uint8)


def add_shadow(optical, alpha, rng):
    """A deliberately mild offset shadow, separate from the cloud-mask target."""
    height, width = alpha.shape[:2]
    shift_x = int(rng.uniform(-0.08, 0.08) * width)
    shift_y = int(rng.uniform(0.04, 0.12) * height)
    transform = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shadow = cv2.warpAffine(alpha.squeeze(-1), transform, (width, height), borderValue=0)
    return optical * (1.0 - 0.20 * shadow[..., None])


def read_binary_mask(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        mask = np.asarray(Image.open(path).convert("L"))
    return (np.asarray(mask).squeeze() > 0).astype(np.uint8)


def real_cloud_layer(source_rgb, source_mask, height, width, rng, opacity):
    """Warp a real cloud texture and its known support onto a target optical image."""
    if rng.random() < 0.5:
        source_rgb = np.fliplr(source_rgb)
        source_mask = np.fliplr(source_mask)
    source_rgb = cv2.resize(source_rgb, (width, height), interpolation=cv2.INTER_CUBIC)
    source_mask = cv2.resize(source_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    angle = float(rng.uniform(-25.0, 25.0))
    scale = float(rng.uniform(0.90, 1.15))
    transform = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    transform[:, 2] += [rng.uniform(-0.12, 0.12) * width, rng.uniform(-0.12, 0.12) * height]
    texture = cv2.warpAffine(source_rgb, transform, (width, height), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REFLECT_101)
    support = cv2.warpAffine(source_mask, transform, (width, height), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    feather_radius = max(2, int(round(min(height, width) * 0.020)))
    feather = Image.fromarray((support > 0).astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(feather_radius))
    support_alpha = np.asarray(feather, dtype=np.float32) / 255.0
    # The automatic support can include terrain beneath translucent clouds.
    # Keep only bright, weakly saturated source pixels as cloud texture so the
    # overlay does not paste roads or water into the target image.
    texture_float = texture.astype(np.float32)
    luminance = (
        0.299 * texture_float[..., 0]
        + 0.587 * texture_float[..., 1]
        + 0.114 * texture_float[..., 2]
    )
    saturation = texture_float.max(axis=-1) - texture_float.min(axis=-1)
    bright = np.clip((luminance - 105.0) / 115.0, 0.0, 1.0)
    neutral = 1.0 - np.clip((saturation - 68.0) / 120.0, 0.0, 1.0)
    cloudness = cv2.GaussianBlur(bright * neutral, (0, 0), sigmaX=1.2)
    alpha = support_alpha * cloudness
    alpha = np.clip(alpha / max(alpha.max(), 1e-6), 0.0, 1.0) * opacity
    texture = 0.85 * texture_float + 0.15 * 255.0
    return texture.astype(np.float32), alpha[..., None], (alpha >= 0.08).astype(np.uint8)


def write_image(array, path):
    Image.fromarray(array.astype(np.uint8)).save(path, quality=95)


def copy_file(source, destination, overwrite):
    if destination.exists() and not overwrite:
        return
    shutil.copy2(source, destination)


def matching_source_file(directory, optical_path):
    """Find an aligned companion file while allowing image/label extensions to differ."""
    candidates = [directory / optical_path.name]
    candidates.extend(directory / f"{optical_path.stem}{suffix}" for suffix in IMAGE_SUFFIXES)
    return next((path for path in candidates if path.is_file()), None)


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    source_opt = source_root / "opt"
    source_sar = source_root / "sar"
    source_lbl = source_root / "lbl"
    if not all(path.is_dir() for path in (source_opt, source_sar, source_lbl)):
        raise FileNotFoundError("--source-root must contain opt/, sar/, and lbl/.")

    selected_stems = read_stems(args.source_list)
    passthrough_stems = read_stems(args.passthrough_list) or set()
    if selected_stems is not None and selected_stems & passthrough_stems:
        overlap = sorted(selected_stems & passthrough_stems)
        raise ValueError(f"--source-list and --passthrough-list overlap: {overlap[:10]}")
    if (args.cloud_source_image is None) != (args.cloud_source_mask is None):
        raise ValueError("Provide both --cloud-source-image and --cloud-source-mask, or neither.")
    if args.backend == "real-patch" and args.cloud_source_image is None:
        raise ValueError("--backend real-patch requires --cloud-source-image and --cloud-source-mask.")
    if args.backend != "real-patch" and args.cloud_source_image is not None:
        raise ValueError("--cloud-source-image/--cloud-source-mask are only used with --backend real-patch.")
    real_cloud_source = None
    if args.cloud_source_image is not None:
        real_cloud_source = (
            np.asarray(Image.open(args.cloud_source_image).convert("RGB")),
            read_binary_mask(args.cloud_source_mask),
        )
    opt_paths = [path for path in sorted(source_opt.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES]
    if selected_stems is not None:
        keep_stems = selected_stems | passthrough_stems
        opt_paths = [path for path in opt_paths if path.stem in keep_stems]
    elif passthrough_stems:
        raise ValueError("--passthrough-list requires --source-list for an explicit mixed split.")
    if args.max_images is not None:
        opt_paths = opt_paths[:args.max_images]
    if not opt_paths:
        raise RuntimeError("No source optical images selected.")

    available_copy_dirs = [name for name in ALIGNED_COPY_DIRS if (source_root / name).is_dir()]
    for directory in [output_root / "opt", output_root / "degradation_gt"] + [
        output_root / name for name in available_copy_dirs
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    severity = dict(SEVERITIES[args.severity])
    if args.coverage is not None:
        severity["coverage"] = tuple(args.coverage)
    if args.opacity is not None:
        severity["opacity"] = tuple(args.opacity)
    if not (0.0 <= severity["coverage"][0] <= severity["coverage"][1] <= 1.0):
        raise ValueError("Coverage bounds must satisfy 0 <= MIN <= MAX <= 1.")
    if not (0.0 <= severity["opacity"][0] <= severity["opacity"][1] <= 1.0):
        raise ValueError("Opacity bounds must satisfy 0 <= MIN <= MAX <= 1.")
    cloud_color = np.asarray(args.cloud_color, dtype=np.float32).reshape(1, 1, 3)
    manifest = []
    synthetic_eval_stems = []
    for index, opt_path in enumerate(tqdm(opt_paths, desc=f"Synthetic {args.severity} clouds")):
        seed = args.seed + index * 1009
        rng = np.random.default_rng(seed)
        optical = np.asarray(Image.open(opt_path).convert("RGB"), dtype=np.float32)
        passthrough = opt_path.stem in passthrough_stems
        coverage = float(rng.uniform(*severity["coverage"])) if not passthrough else None
        opacity = float(rng.uniform(*severity["opacity"])) if not passthrough else None
        if passthrough:
            corrupted = optical
            gt_mask = np.zeros(optical.shape[:2], dtype=np.uint8)
        elif args.backend == "multiscale":
            alpha, gt_mask = multiscale_cloud_field(
                optical.shape[0], optical.shape[1], rng, coverage, opacity, args.opaque_core_strength
            )
            # Slight sample-to-sample color variation avoids uniformly white overlays.
            cloud_layer = np.clip(cloud_color + rng.normal(0.0, 4.0, size=(1, 1, 3)), 210.0, 255.0)
            corrupted = optical * (1.0 - alpha) + cloud_layer * alpha
        elif args.backend == "scg":
            corrupted, alpha, gt_mask = scg_cloud_field(optical, opacity, args)
        else:
            cloud_layer, alpha, gt_mask = real_cloud_layer(
                *real_cloud_source, optical.shape[0], optical.shape[1], rng, opacity
            )
            corrupted = optical * (1.0 - alpha) + cloud_layer * alpha
        if args.shadow and not passthrough:
            corrupted = add_shadow(corrupted, alpha, rng)

        if passthrough:
            # Preserve naturally degraded observations bit-for-bit; they are
            # part of the real test distribution, not a synthetic rendering.
            copy_file(opt_path, output_root / "opt" / opt_path.name, args.overwrite)
        else:
            write_image(corrupted, output_root / "opt" / opt_path.name)
        Image.fromarray(gt_mask * 255).save(output_root / "degradation_gt" / f"{opt_path.stem}.png")
        for name in available_copy_dirs:
            source_file = matching_source_file(source_root / name, opt_path)
            if source_file is None:
                raise FileNotFoundError(f"Missing aligned {name} file for {opt_path.name}.")
            copy_file(source_file, output_root / name / source_file.name, args.overwrite)
        manifest.append({
            "id": opt_path.stem,
            "seed": seed,
            "passthrough_natural_cloud": passthrough,
            "severity": args.severity,
            "target_coverage": coverage,
            "opacity": opacity,
            "opaque_core_strength": args.opaque_core_strength,
            "actual_mask_ratio": float(gt_mask.mean()) if not passthrough else None,
            "backend": args.backend,
            "cloud_source": str(args.cloud_source_image) if args.cloud_source_image else None,
            "shadow": args.shadow,
        })

        if not passthrough:
            synthetic_eval_stems.append(opt_path.stem)

    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_root / "synthetic_mask_eval_ids.txt").write_text(
        "\n".join(synthetic_eval_stems) + "\n", encoding="utf-8"
    )
    protocol = {
        "source_root": str(source_root),
        "severity": args.severity,
        "seed": args.seed,
        "backend": args.backend,
        "shadow": args.shadow,
        "opaque_core_strength": args.opaque_core_strength,
        "copied_aligned_directories": available_copy_dirs,
        "not_copied": "SAM/Felzenszwalb proposal directories; regenerate them from output_root/opt.",
        "num_images": len(manifest),
        "num_passthrough_natural_cloud": len(passthrough_stems),
        "synthetic_mask_eval_ids": "synthetic_mask_eval_ids.txt; only these samples have known degradation masks.",
        "mask_usage": "evaluation only; never feed degradation_gt to the model. "
                      "Do not evaluate natural-cloud passthrough samples against their all-zero placeholder masks.",
        "required_next_step": "Regenerate region proposals from output_root/opt before model inference.",
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Created {len(manifest)} images in {output_root}")


if __name__ == "__main__":
    main()
