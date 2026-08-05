from __future__ import annotations

import argparse
import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import sys

import cv2
import numpy as np
import pandas as pd
import rasterio
import vismatch

# Workaround: vismatch vendored third-party repos (MINIMA, XoFTR,
# MatchAnything) each ship their own ``src/`` package.  We do NOT pre-add them
# to sys.path here because they conflict; instead we purge the cached ``src``
# module between matchers (see the loop below) so each matcher's own
# ``add_to_path`` call wins.
from rsim.matcher_roster import DEFAULT_MATCHERS


@dataclass(frozen=True)
class PairSample:
    case_id: str
    pair_id: str
    optical_path: Path
    sar_path: Path
    tiepoints_path: Path


@dataclass(frozen=True)
class TileModel:
    sar_x0: int
    sar_y0: int
    sar_x1: int
    sar_y1: int
    center_x: float
    center_y: float
    homography: np.ndarray
    inliers: int
    matches: int
    inlier_ratio: float
    spatial_coverage: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/spacenet9/train"))
    parser.add_argument(
        "--matchers",
        nargs="+",
        default=DEFAULT_MATCHERS,
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--bf16-on-oom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry CUDA OOM inference in bfloat16 autocast.",
    )
    parser.add_argument("--max-num-keypoints", type=int, default=4096)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "tiled"],
        default="tiled",
    )
    parser.add_argument(
        "--normalizations",
        nargs="+",
        default=["identity", "percentile", "zscore"],
        help="Normalization methods to sweep.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="Resize long image side to this value before matching (0 keeps original size).",
    )
    parser.add_argument(
        "--fallback-max-side",
        type=int,
        default=0,
        help="Retry with this resize if inference fails (0 disables fallback).",
    )
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=256)
    parser.add_argument("--min-tile-inliers", type=int, default=8)
    parser.add_argument(
        "--geometry-model",
        type=str,
        choices=["homography", "affine"],
        default="homography",
        help="Geometric model used to map matched keypoints.",
    )
    parser.add_argument(
        "--ransac-reproj-threshold",
        type=float,
        default=3.0,
        help="RANSAC reprojection threshold (pixels) for geometric model fitting.",
    )
    parser.add_argument(
        "--coverage-grid-size",
        type=int,
        default=4,
        help="Grid size used for spatial coverage diagnostics.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Number of discovered pairs to evaluate (0 uses all discovered pairs).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for deterministic pair sampling and bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--sample-by-case",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When sampling, interleave selection across case IDs before exhausting any case.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-ci",
        type=float,
        default=95.0,
        help="Bootstrap confidence interval level in percent.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/spacenet9_train_matcher_results.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/spacenet9_train_matcher_summary.csv"),
    )
    return parser.parse_args()


def parse_matcher_spec(spec: str) -> str | list[str]:
    if "+" not in spec:
        return spec
    parts = [item.strip() for item in spec.split("+") if item.strip()]
    if len(parts) < 2:
        return spec
    return parts


def discover_pairs(data_dir: Path) -> list[PairSample]:
    pattern = re.compile(r"^(\d+)_optical_train_(\d+)\.tif$")
    pairs: list[PairSample] = []

    for optical_path in sorted(data_dir.glob("*_optical_train_*.tif")):
        match = pattern.match(optical_path.name)
        if match is None:
            continue
        case_id, pair_id = match.groups()
        sar_path = data_dir / f"{case_id}_sar_train_{pair_id}.tif"
        tiepoints_path = data_dir / f"{case_id}_tiepoints_train_{pair_id}.csv"
        if sar_path.exists() and tiepoints_path.exists():
            pairs.append(
                PairSample(
                    case_id=case_id,
                    pair_id=pair_id,
                    optical_path=optical_path,
                    sar_path=sar_path,
                    tiepoints_path=tiepoints_path,
                )
            )
    if not pairs:
        msg = f"No valid SpaceNet9 train pairs found under {data_dir}."
        raise FileNotFoundError(msg)
    return pairs


class _HasCaseId(Protocol):
    # Read-only property so frozen dataclasses (read-only attributes) satisfy
    # the protocol; a bare ``case_id: str`` implies a mutable attribute.
    @property
    def case_id(self) -> str: ...


def sample_pairs[PairT: _HasCaseId](
    pairs: list[PairT],
    sample_size: int,
    sample_seed: int,
    sample_by_case: bool,
) -> list[PairT]:
    if sample_size <= 0 or sample_size >= len(pairs):
        return pairs

    rng = np.random.default_rng(sample_seed)
    if not sample_by_case:
        selected_idx = sorted(rng.choice(len(pairs), size=sample_size, replace=False).tolist())
        return [pairs[index] for index in selected_idx]

    case_to_indices: dict[str, list[int]] = {}
    for index, pair in enumerate(pairs):
        case_to_indices.setdefault(pair.case_id, []).append(index)

    shuffled_buckets: dict[str, list[int]] = {}
    for case_id, indices in case_to_indices.items():
        shuffled_indices = indices.copy()
        rng.shuffle(shuffled_indices)
        shuffled_buckets[case_id] = shuffled_indices

    selected: list[int] = []
    ordered_cases = sorted(shuffled_buckets.keys())
    while len(selected) < sample_size:
        progressed = False
        for case_id in ordered_cases:
            bucket = shuffled_buckets[case_id]
            if not bucket:
                continue
            selected.append(bucket.pop())
            progressed = True
            if len(selected) >= sample_size:
                break
        if not progressed:
            break

    selected.sort()
    return [pairs[index] for index in selected]


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    bootstrap_samples: int,
    confidence_level: float,
) -> tuple[float, float]:
    valid_values = values[np.isfinite(values)]
    if valid_values.size == 0:
        return float("nan"), float("nan")
    if valid_values.size == 1:
        single = float(valid_values[0])
        return single, single

    sample_count = max(1, int(bootstrap_samples))
    means = np.empty(sample_count, dtype=np.float64)
    n = valid_values.size
    for i in range(sample_count):
        sample_indices = rng.integers(0, n, size=n)
        means[i] = float(np.mean(valid_values[sample_indices]))

    clipped_confidence = float(np.clip(confidence_level, 0.0, 100.0))
    alpha = (100.0 - clipped_confidence) / 2.0
    low = float(np.percentile(means, alpha))
    high = float(np.percentile(means, 100.0 - alpha))
    return low, high


def group_bootstrap_ci_row(
    group: pd.DataFrame,
    bootstrap_samples: int,
    confidence_level: float,
    sample_seed: int,
) -> dict[str, float]:
    matcher = str(group["matcher"].iloc[0])
    normalization = str(group["normalization"].iloc[0])
    mode = str(group["mode"].iloc[0])
    group_fingerprint = f"{matcher}|{normalization}|{mode}"
    group_hash = int.from_bytes(
        hashlib.sha256(group_fingerprint.encode("utf-8")).digest()[:8], "little"
    )
    rng = np.random.default_rng(sample_seed + group_hash)

    is_failure = (~group["status"].astype(str).str.startswith("ok")).to_numpy(dtype=np.float64)
    mean_error = group["mean_error_px"].to_numpy(dtype=np.float64)
    success_5 = group["success_at_5px"].to_numpy(dtype=np.float64)
    success_10 = group["success_at_10px"].to_numpy(dtype=np.float64)

    failure_low, failure_high = bootstrap_mean_ci(
        is_failure,
        rng=rng,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
    )
    mean_error_low, mean_error_high = bootstrap_mean_ci(
        mean_error,
        rng=rng,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
    )
    success_5_low, success_5_high = bootstrap_mean_ci(
        success_5,
        rng=rng,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
    )
    success_10_low, success_10_high = bootstrap_mean_ci(
        success_10,
        rng=rng,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
    )

    return {
        "failure_rate_ci_low": failure_low,
        "failure_rate_ci_high": failure_high,
        "mean_error_px_ci_low": mean_error_low,
        "mean_error_px_ci_high": mean_error_high,
        "success_at_5px_ci_low": success_5_low,
        "success_at_5px_ci_high": success_5_high,
        "success_at_10px_ci_low": success_10_low,
        "success_at_10px_ci_high": success_10_high,
    }


def load_tiepoints(tiepoints_path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(tiepoints_path)
    required = {"sar_row", "sar_col", "optical_row", "optical_col"}
    missing = required.difference(frame.columns)
    if missing:
        msg = f"Missing columns in {tiepoints_path}: {sorted(missing)}"
        raise ValueError(msg)
    sar = frame[["sar_col", "sar_row"]].to_numpy(dtype=np.float64)
    optical = frame[["optical_col", "optical_row"]].to_numpy(dtype=np.float64)
    return sar, optical


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32)
    value_min = float(np.nanmin(image))
    value_max = float(np.nanmax(image))
    if value_max <= value_min:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - value_min) / (value_max - value_min)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def read_image(path: Path) -> np.ndarray:
    with rasterio.open(path) as dataset:
        image = dataset.read()
    image = np.transpose(image, (1, 2, 0))
    image = _normalize_to_uint8(image)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return image


def normalize_image(image: np.ndarray, method: str, modality: str) -> np.ndarray:
    image_float = image.astype(np.float32)

    def grayscale_three_channel(image_uint8: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        return np.repeat(gray[:, :, None], 3, axis=2)

    def minmax_percentile(
        channel_data: np.ndarray, low: float = 2.0, high: float = 98.0
    ) -> np.ndarray:
        p_low = float(np.percentile(channel_data, low))
        p_high = float(np.percentile(channel_data, high))
        if p_high <= p_low:
            return np.clip(channel_data / 255.0, 0.0, 1.0)
        return np.clip((channel_data - p_low) / (p_high - p_low), 0.0, 1.0)

    if method == "identity":
        return np.clip(image_float / 255.0, 0.0, 1.0)

    if method == "percentile":
        output = np.empty_like(image_float, dtype=np.float32)
        for channel in range(image_float.shape[2]):
            output[:, :, channel] = minmax_percentile(
                image_float[:, :, channel], low=2.0, high=98.0
            )
        return output

    if method == "sarlog_percentile":
        output = np.empty_like(image_float, dtype=np.float32)
        for channel in range(image_float.shape[2]):
            channel_data = image_float[:, :, channel]
            if modality == "sar":
                channel_data = np.log1p(np.clip(channel_data, a_min=0.0, a_max=None))
            output[:, :, channel] = minmax_percentile(channel_data, low=2.0, high=98.0)
        return output

    if method == "winsol_gray_blur":
        image_uint8 = image.astype(np.uint8)
        if modality == "optical":
            gray_three = grayscale_three_channel(image_uint8)
            return gray_three.astype(np.float32) / 255.0
        blurred = cv2.GaussianBlur(image_uint8, (7, 7), 0)
        return blurred.astype(np.float32) / 255.0

    if method == "winsol_gray_bilat_histeq_blur":
        image_uint8 = image.astype(np.uint8)
        if modality == "optical":
            gray_three = grayscale_three_channel(image_uint8)
            return gray_three.astype(np.float32) / 255.0

        output_uint8 = np.empty_like(image_uint8, dtype=np.uint8)
        for channel in range(image_uint8.shape[2]):
            filtered = cv2.bilateralFilter(
                image_uint8[:, :, channel], d=9, sigmaColor=75, sigmaSpace=75
            )
            equalized = cv2.equalizeHist(filtered)
            output_uint8[:, :, channel] = cv2.GaussianBlur(equalized, (5, 5), 0)
        return output_uint8.astype(np.float32) / 255.0

    if method == "winsol_gray_bilat_clahe_blur_sharp":
        image_uint8 = image.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if modality == "optical":
            gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
            enhanced = clahe.apply(gray)
            gray_three = np.repeat(enhanced[:, :, None], 3, axis=2)
            return gray_three.astype(np.float32) / 255.0

        sharpen_kernel = np.array(
            [
                [0.0, -1.0, 0.0],
                [-1.0, 5.0, -1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
        output_uint8 = np.empty_like(image_uint8, dtype=np.uint8)
        for channel in range(image_uint8.shape[2]):
            filtered = cv2.bilateralFilter(
                image_uint8[:, :, channel], d=9, sigmaColor=75, sigmaSpace=75
            )
            enhanced = clahe.apply(filtered)
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            output_uint8[:, :, channel] = cv2.filter2D(blurred, ddepth=-1, kernel=sharpen_kernel)
        return output_uint8.astype(np.float32) / 255.0

    if method == "winsol_sarlog_p2p98_masked":
        output = np.empty_like(image_float, dtype=np.float32)
        for channel in range(image_float.shape[2]):
            channel_data = image_float[:, :, channel]
            if modality == "sar":
                valid = channel_data > 0.0
                transformed = np.log1p(np.clip(channel_data, a_min=0.0, a_max=None))
                if np.any(valid):
                    valid_values = transformed[valid]
                    p_low = float(np.percentile(valid_values, 2.0))
                    p_high = float(np.percentile(valid_values, 98.0))
                    if p_high <= p_low:
                        output[:, :, channel] = np.clip(transformed / 255.0, 0.0, 1.0)
                    else:
                        output[:, :, channel] = np.clip(
                            (transformed - p_low) / (p_high - p_low), 0.0, 1.0
                        )
                else:
                    output[:, :, channel] = minmax_percentile(transformed, low=2.0, high=98.0)
            else:
                output[:, :, channel] = minmax_percentile(channel_data, low=2.0, high=98.0)
        return output

    if method == "zscore":
        output = np.empty_like(image_float, dtype=np.float32)
        for channel in range(image_float.shape[2]):
            channel_data = image_float[:, :, channel]
            mean = float(np.mean(channel_data))
            std = float(np.std(channel_data))
            if std < 1e-6:
                output[:, :, channel] = 0.5
            else:
                standardized = (channel_data - mean) / std
                standardized = np.clip(standardized, -2.5, 2.5)
                output[:, :, channel] = (standardized + 2.5) / 5.0
        return output

    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        output_uint8 = np.empty_like(image, dtype=np.uint8)
        for channel in range(image.shape[2]):
            output_uint8[:, :, channel] = clahe.apply(image[:, :, channel])
        return output_uint8.astype(np.float32) / 255.0

    msg = f"Unknown normalization method: {method}"
    raise ValueError(msg)


def resize_to_max_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if max_side <= 0 or longest <= max_side:
        return image, 1.0, 1.0
    scale = max_side / float(longest)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    sx = new_width / float(width)
    sy = new_height / float(height)
    return resized, sx, sy


def tiled_starts(length: int, tile_size: int, tile_overlap: int) -> list[int]:
    stride = max(1, tile_size - tile_overlap)
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    final_start = max(0, length - tile_size)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return sorted(set(starts))


def apply_homography(points_xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
    ones = np.ones((points_xy.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate([points_xy, ones], axis=1)
    mapped = (homography @ homogeneous.T).T
    return mapped[:, :2] / np.clip(mapped[:, 2:3], a_min=1e-9, a_max=None)


def fit_geometric_model(
    matched_kpts0: np.ndarray,
    matched_kpts1: np.ndarray,
    geometry_model: str,
    ransac_reproj_threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if matched_kpts0.shape[0] != matched_kpts1.shape[0] or matched_kpts0.shape[0] == 0:
        return None, None

    points0 = matched_kpts0.astype(np.float64)
    points1 = matched_kpts1.astype(np.float64)

    if geometry_model == "homography":
        if points0.shape[0] < 4:
            return None, None
        model, mask = cv2.findHomography(
            points0,
            points1,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_reproj_threshold),
        )
        if model is None:
            return None, None
        return np.asarray(model, dtype=np.float64), mask

    if geometry_model == "affine":
        if points0.shape[0] < 3:
            return None, None
        affine_model, mask = cv2.estimateAffine2D(
            points0,
            points1,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_reproj_threshold),
        )
        if affine_model is None:
            return None, None
        full_model = np.eye(3, dtype=np.float64)
        full_model[:2, :] = np.asarray(affine_model, dtype=np.float64)
        return full_model, mask

    msg = f"Unknown geometry model: {geometry_model}"
    raise ValueError(msg)


def spatial_coverage_score(
    points_xy: np.ndarray,
    width: int,
    height: int,
    grid_size: int,
) -> float:
    if points_xy.size == 0 or width <= 0 or height <= 0:
        return 0.0
    grid = max(1, int(grid_size))
    x_norm = np.clip(points_xy[:, 0] / float(width), 0.0, 0.999999)
    y_norm = np.clip(points_xy[:, 1] / float(height), 0.0, 0.999999)
    x_bins = np.floor(x_norm * grid).astype(np.int32)
    y_bins = np.floor(y_norm * grid).astype(np.int32)
    occupied = set(zip(x_bins.tolist(), y_bins.tolist(), strict=False))
    return float(len(occupied) / float(grid * grid))


def to_matcher_tensor(image: np.ndarray, normalization: str, modality: str) -> np.ndarray:
    normalized = normalize_image(image, normalization, modality)
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32)


def empty_metric_row(status: str, elapsed: float = float("nan")) -> dict[str, Any]:
    return {
        "status": status,
        "num_inliers": 0,
        "num_matches": 0,
        "inlier_ratio": float("nan"),
        "spatial_coverage": float("nan"),
        "mean_error_px": float("nan"),
        "median_error_px": float("nan"),
        "success_at_1px": float("nan"),
        "success_at_2px": float("nan"),
        "success_at_3px": float("nan"),
        "success_at_5px": float("nan"),
        "success_at_10px": float("nan"),
        "mean_elapsed_sec": elapsed,
        "tiles_used": 0,
        "tile_trials": 0,
        "precision_mode": "none",
    }


def is_cuda_oom(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "out of memory" in message
        or "cuda error: out of memory" in message
        or "cudnn_status_alloc_failed" in message
    )


@torch.inference_mode()
def run_matcher_with_precision(
    matcher: vismatch.BaseMatcher,
    sar_tensor: np.ndarray,
    optical_tensor: np.ndarray,
    device: str,
    bf16_on_oom: bool,
) -> tuple[dict[str, Any], str]:
    try:
        return matcher(sar_tensor, optical_tensor), "fp32"
    except RuntimeError as error:
        if not (bf16_on_oom and device.startswith("cuda") and is_cuda_oom(error)):
            raise
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise
        torch.cuda.empty_cache()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = matcher(sar_tensor, optical_tensor)
        return result, "bf16"


def compute_success_metrics(errors: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for threshold in [1, 2, 3, 5, 10]:
        result[f"success_at_{threshold}px"] = float(np.mean(errors <= threshold))
    return result


def build_metric_row(
    status: str,
    errors: np.ndarray,
    num_inliers: int,
    num_matches: int,
    inlier_ratio: float,
    spatial_coverage: float,
    elapsed_seconds: float,
    tiles_used: int,
    tile_trials: int,
    precision_mode: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "num_inliers": num_inliers,
        "num_matches": num_matches,
        "inlier_ratio": inlier_ratio,
        "spatial_coverage": spatial_coverage,
        "mean_error_px": float(np.mean(errors)),
        "median_error_px": float(np.median(errors)),
        **compute_success_metrics(errors),
        "mean_elapsed_sec": elapsed_seconds,
        "tiles_used": tiles_used,
        "tile_trials": tile_trials,
        "precision_mode": precision_mode,
    }


def evaluate_full_homography(
    matcher: vismatch.BaseMatcher,
    sar_image: np.ndarray,
    optical_image: np.ndarray,
    sar_tiepoints_xy: np.ndarray,
    optical_tiepoints_xy: np.ndarray,
    normalization: str,
    geometry_model: str,
    ransac_reproj_threshold: float,
    coverage_grid_size: int,
    device: str,
    bf16_on_oom: bool,
) -> dict[str, Any]:
    sar_tensor = to_matcher_tensor(sar_image, normalization, modality="sar")
    optical_tensor = to_matcher_tensor(optical_image, normalization, modality="optical")

    start_time = time.perf_counter()
    result, precision_mode = run_matcher_with_precision(
        matcher=matcher,
        sar_tensor=sar_tensor,
        optical_tensor=optical_tensor,
        device=device,
        bf16_on_oom=bf16_on_oom,
    )
    elapsed_seconds = time.perf_counter() - start_time

    matched_kpts0 = np.asarray(result.get("matched_kpts0", np.empty((0, 2))), dtype=np.float64)
    matched_kpts1 = np.asarray(result.get("matched_kpts1", np.empty((0, 2))), dtype=np.float64)
    num_matches = int(matched_kpts0.shape[0])
    h_resized, inlier_mask = fit_geometric_model(
        matched_kpts0,
        matched_kpts1,
        geometry_model=geometry_model,
        ransac_reproj_threshold=ransac_reproj_threshold,
    )
    num_inliers = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
    inlier_ratio = float(num_inliers / max(1, num_matches))
    inlier_points = (
        matched_kpts0[inlier_mask.ravel() > 0] if inlier_mask is not None else np.empty((0, 2))
    )
    coverage = spatial_coverage_score(
        inlier_points,
        width=sar_image.shape[1],
        height=sar_image.shape[0],
        grid_size=coverage_grid_size,
    )

    if h_resized is None:
        return empty_metric_row("no_homography", elapsed_seconds)

    predicted_optical_xy = apply_homography(
        sar_tiepoints_xy,
        np.asarray(h_resized, dtype=np.float64),
    )
    if not np.isfinite(predicted_optical_xy).all():
        return empty_metric_row("invalid_projection", elapsed_seconds)

    errors = np.linalg.norm(predicted_optical_xy - optical_tiepoints_xy, axis=1)
    if float(np.mean(errors)) > 1e6:
        return empty_metric_row("unstable_homography", elapsed_seconds)

    return build_metric_row(
        status="ok",
        errors=errors,
        num_inliers=num_inliers,
        num_matches=num_matches,
        inlier_ratio=inlier_ratio,
        spatial_coverage=coverage,
        elapsed_seconds=elapsed_seconds,
        tiles_used=1,
        tile_trials=1,
        precision_mode=precision_mode,
    )


def tile_models(
    matcher: vismatch.BaseMatcher,
    sar_image: np.ndarray,
    optical_image: np.ndarray,
    normalization: str,
    tile_size: int,
    tile_overlap: int,
    min_tile_inliers: int,
    geometry_model: str,
    ransac_reproj_threshold: float,
    coverage_grid_size: int,
    device: str,
    bf16_on_oom: bool,
) -> tuple[list[TileModel], float, int, int]:
    sar_h, sar_w = sar_image.shape[:2]
    opt_h, opt_w = optical_image.shape[:2]
    x_starts = tiled_starts(sar_w, tile_size, tile_overlap)
    y_starts = tiled_starts(sar_h, tile_size, tile_overlap)

    ratio_x = opt_w / float(sar_w)
    ratio_y = opt_h / float(sar_h)

    models: list[TileModel] = []
    total_elapsed = 0.0
    bf16_calls = 0
    tile_trials = len(x_starts) * len(y_starts)

    for sar_y0 in y_starts:
        for sar_x0 in x_starts:
            sar_x1 = min(sar_w, sar_x0 + tile_size)
            sar_y1 = min(sar_h, sar_y0 + tile_size)
            sar_tile = sar_image[sar_y0:sar_y1, sar_x0:sar_x1]

            sar_center_x = sar_x0 + (sar_x1 - sar_x0) / 2.0
            sar_center_y = sar_y0 + (sar_y1 - sar_y0) / 2.0

            opt_tile_w = max(64, round((sar_x1 - sar_x0) * ratio_x))
            opt_tile_h = max(64, round((sar_y1 - sar_y0) * ratio_y))
            opt_center_x = sar_center_x * ratio_x
            opt_center_y = sar_center_y * ratio_y

            opt_x0 = round(opt_center_x - opt_tile_w / 2.0)
            opt_y0 = round(opt_center_y - opt_tile_h / 2.0)
            opt_x0 = max(0, min(opt_x0, max(0, opt_w - opt_tile_w)))
            opt_y0 = max(0, min(opt_y0, max(0, opt_h - opt_tile_h)))
            opt_x1 = min(opt_w, opt_x0 + opt_tile_w)
            opt_y1 = min(opt_h, opt_y0 + opt_tile_h)
            optical_tile = optical_image[opt_y0:opt_y1, opt_x0:opt_x1]

            if sar_tile.size == 0 or optical_tile.size == 0:
                continue

            sar_tensor = to_matcher_tensor(sar_tile, normalization, modality="sar")
            optical_tensor = to_matcher_tensor(optical_tile, normalization, modality="optical")

            start_time = time.perf_counter()
            try:
                result, precision_mode = run_matcher_with_precision(
                    matcher=matcher,
                    sar_tensor=sar_tensor,
                    optical_tensor=optical_tensor,
                    device=device,
                    bf16_on_oom=bf16_on_oom,
                )
                if precision_mode == "bf16":
                    bf16_calls += 1
            except Exception:
                total_elapsed += time.perf_counter() - start_time
                continue
            total_elapsed += time.perf_counter() - start_time

            matched = np.asarray(result.get("matched_kpts0", np.empty((0, 2))), dtype=np.float64)
            matched_target = np.asarray(
                result.get("matched_kpts1", np.empty((0, 2))), dtype=np.float64
            )
            matches = int(matched.shape[0])
            h_tile, inlier_mask = fit_geometric_model(
                matched,
                matched_target,
                geometry_model=geometry_model,
                ransac_reproj_threshold=ransac_reproj_threshold,
            )
            inliers = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
            inlier_ratio = float(inliers / max(1, matches))
            inlier_points = (
                matched[inlier_mask.ravel() > 0] if inlier_mask is not None else np.empty((0, 2))
            )
            coverage = spatial_coverage_score(
                inlier_points,
                width=max(1, sar_x1 - sar_x0),
                height=max(1, sar_y1 - sar_y0),
                grid_size=coverage_grid_size,
            )
            if h_tile is None or inliers < min_tile_inliers:
                continue

            t_sar = np.array(
                [
                    [1.0, 0.0, sar_x0],
                    [0.0, 1.0, sar_y0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            t_opt = np.array(
                [
                    [1.0, 0.0, opt_x0],
                    [0.0, 1.0, opt_y0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            full_h = t_opt @ np.asarray(h_tile, dtype=np.float64) @ np.linalg.inv(t_sar)
            if not np.isfinite(full_h).all():
                continue

            models.append(
                TileModel(
                    sar_x0=sar_x0,
                    sar_y0=sar_y0,
                    sar_x1=sar_x1,
                    sar_y1=sar_y1,
                    center_x=sar_center_x,
                    center_y=sar_center_y,
                    homography=full_h,
                    inliers=inliers,
                    matches=matches,
                    inlier_ratio=inlier_ratio,
                    spatial_coverage=coverage,
                )
            )

    return models, total_elapsed, bf16_calls, tile_trials


def predict_with_tile_models(
    sar_tiepoints_xy: np.ndarray, models: list[TileModel]
) -> np.ndarray | None:
    if not models:
        return None

    predictions: list[np.ndarray] = []
    for point in sar_tiepoints_xy:
        x, y = float(point[0]), float(point[1])
        candidates = [
            model
            for model in models
            if model.sar_x0 <= x < model.sar_x1 and model.sar_y0 <= y < model.sar_y1
        ]
        if not candidates:
            candidates = models

        weighted_points: list[tuple[np.ndarray, float]] = []
        for model in candidates:
            projected = apply_homography(np.array([[x, y]], dtype=np.float64), model.homography)[0]
            if not np.isfinite(projected).all():
                continue
            distance = np.hypot(x - model.center_x, y - model.center_y)
            weight = (1.0 / (1.0 + distance)) * max(1.0, float(model.inliers))
            weighted_points.append((projected, weight))

        if not weighted_points:
            return None

        projected_points = np.stack([projected for projected, _ in weighted_points], axis=0)
        weights = np.asarray([weight for _, weight in weighted_points], dtype=np.float64)
        blended = np.average(projected_points, axis=0, weights=weights)
        predictions.append(blended)

    return np.vstack(predictions)


def evaluate_tiled_homography(
    matcher: vismatch.BaseMatcher,
    sar_image: np.ndarray,
    optical_image: np.ndarray,
    sar_tiepoints_xy: np.ndarray,
    optical_tiepoints_xy: np.ndarray,
    tile_size: int,
    tile_overlap: int,
    min_tile_inliers: int,
    normalization: str,
    geometry_model: str,
    ransac_reproj_threshold: float,
    coverage_grid_size: int,
    device: str,
    bf16_on_oom: bool,
) -> dict[str, Any]:
    models, elapsed_seconds, bf16_calls, tile_trials = tile_models(
        matcher=matcher,
        sar_image=sar_image,
        optical_image=optical_image,
        normalization=normalization,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        min_tile_inliers=min_tile_inliers,
        geometry_model=geometry_model,
        ransac_reproj_threshold=ransac_reproj_threshold,
        coverage_grid_size=coverage_grid_size,
        device=device,
        bf16_on_oom=bf16_on_oom,
    )

    total_inliers = int(sum(model.inliers for model in models))
    total_matches = int(sum(model.matches for model in models))
    global_inlier_ratio = float(total_inliers / max(1, total_matches))
    mean_spatial_coverage = float(np.mean([model.spatial_coverage for model in models]))

    if not models:
        return empty_metric_row("no_tiles", elapsed_seconds)

    predictions = predict_with_tile_models(sar_tiepoints_xy, models)
    if predictions is None or not np.isfinite(predictions).all():
        return empty_metric_row("invalid_projection", elapsed_seconds)

    errors = np.linalg.norm(predictions - optical_tiepoints_xy, axis=1)
    if float(np.mean(errors)) > 1e6:
        return empty_metric_row("unstable_homography", elapsed_seconds)

    return build_metric_row(
        status="ok",
        errors=errors,
        num_inliers=total_inliers,
        num_matches=total_matches,
        inlier_ratio=global_inlier_ratio,
        spatial_coverage=mean_spatial_coverage,
        elapsed_seconds=elapsed_seconds,
        tiles_used=len(models),
        tile_trials=tile_trials,
        precision_mode="bf16" if bf16_calls > 0 else "fp32",
    )


def evaluate_pair(
    matcher: vismatch.BaseMatcher,
    mode: str,
    sar_image: np.ndarray,
    optical_image: np.ndarray,
    sar_tiepoints_xy: np.ndarray,
    optical_tiepoints_xy: np.ndarray,
    normalization: str,
    tile_size: int,
    tile_overlap: int,
    min_tile_inliers: int,
    geometry_model: str,
    ransac_reproj_threshold: float,
    coverage_grid_size: int,
    device: str,
    bf16_on_oom: bool,
) -> dict[str, Any]:
    if mode == "full":
        return evaluate_full_homography(
            matcher=matcher,
            sar_image=sar_image,
            optical_image=optical_image,
            sar_tiepoints_xy=sar_tiepoints_xy,
            optical_tiepoints_xy=optical_tiepoints_xy,
            normalization=normalization,
            geometry_model=geometry_model,
            ransac_reproj_threshold=ransac_reproj_threshold,
            coverage_grid_size=coverage_grid_size,
            device=device,
            bf16_on_oom=bf16_on_oom,
        )

    return evaluate_tiled_homography(
        matcher=matcher,
        sar_image=sar_image,
        optical_image=optical_image,
        sar_tiepoints_xy=sar_tiepoints_xy,
        optical_tiepoints_xy=optical_tiepoints_xy,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        min_tile_inliers=min_tile_inliers,
        normalization=normalization,
        geometry_model=geometry_model,
        ransac_reproj_threshold=ransac_reproj_threshold,
        coverage_grid_size=coverage_grid_size,
        device=device,
        bf16_on_oom=bf16_on_oom,
    )


def run_experiments(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    vismatch.disable_progress_bars()
    discovered_pairs = discover_pairs(args.data_dir)
    pairs = sample_pairs(
        discovered_pairs,
        sample_size=int(args.sample_size),
        sample_seed=int(args.sample_seed),
        sample_by_case=bool(args.sample_by_case),
    )

    records: list[dict[str, Any]] = []
    image_cache: dict[Path, np.ndarray] = {}

    for matcher_name in args.matchers:
        matcher_spec = parse_matcher_spec(matcher_name)
        # Purge cached ``src`` package between matchers: multiple vendored
        # third-party repos (MINIMA, XoFTR, MatchAnything) each ship their own
        # ``src/`` namespace and the cached one may not match the next matcher.
        # Also remove their paths from sys.path so the next matcher's
        # ``add_to_path`` call wins.
        for _key in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[_key]
        _tp = str(Path(vismatch.__file__).parent / "third_party")
        sys.path[:] = [p for p in sys.path if not p.startswith(_tp)]
        try:
            matcher = vismatch.get_matcher(
                matcher_name=matcher_spec,
                device=args.device,
                max_num_keypoints=args.max_num_keypoints,
            )
        except Exception as init_error:
            for normalization in args.normalizations:
                for pair in pairs:
                    tiepoint_count = int(load_tiepoints(pair.tiepoints_path)[0].shape[0])
                    records.append(
                        {
                            "matcher": matcher_name,
                            "normalization": normalization,
                            "mode": args.mode,
                            "case_id": pair.case_id,
                            "pair_id": pair.pair_id,
                            "tiepoints": tiepoint_count,
                            "max_side": int(args.max_side),
                            "sample_size": int(args.sample_size),
                            "sample_seed": int(args.sample_seed),
                            "tile_size": int(args.tile_size),
                            "tile_overlap": int(args.tile_overlap),
                            "geometry_model": str(args.geometry_model),
                            **empty_metric_row(f"init_error:{type(init_error).__name__}"),
                        }
                    )
            continue

        for normalization in args.normalizations:
            for pair in pairs:
                sar_tiepoints_xy, optical_tiepoints_xy = load_tiepoints(pair.tiepoints_path)
                if pair.sar_path not in image_cache:
                    image_cache[pair.sar_path] = read_image(pair.sar_path)
                if pair.optical_path not in image_cache:
                    image_cache[pair.optical_path] = read_image(pair.optical_path)

                sar_image = image_cache[pair.sar_path]
                optical_image = image_cache[pair.optical_path]

                resize_value = args.max_side
                sar_resized, sar_sx, sar_sy = resize_to_max_side(sar_image, resize_value)
                optical_resized, opt_sx, opt_sy = resize_to_max_side(optical_image, resize_value)

                sar_tiepoints_scaled = sar_tiepoints_xy.copy()
                sar_tiepoints_scaled[:, 0] *= sar_sx
                sar_tiepoints_scaled[:, 1] *= sar_sy
                optical_tiepoints_scaled = optical_tiepoints_xy.copy()
                optical_tiepoints_scaled[:, 0] *= opt_sx
                optical_tiepoints_scaled[:, 1] *= opt_sy

                try:
                    metric_row = evaluate_pair(
                        matcher=matcher,
                        mode=args.mode,
                        sar_image=sar_resized,
                        optical_image=optical_resized,
                        sar_tiepoints_xy=sar_tiepoints_scaled,
                        optical_tiepoints_xy=optical_tiepoints_scaled,
                        normalization=normalization,
                        tile_size=args.tile_size,
                        tile_overlap=args.tile_overlap,
                        min_tile_inliers=args.min_tile_inliers,
                        geometry_model=args.geometry_model,
                        ransac_reproj_threshold=args.ransac_reproj_threshold,
                        coverage_grid_size=args.coverage_grid_size,
                        device=args.device,
                        bf16_on_oom=args.bf16_on_oom,
                    )
                except Exception as error:
                    if args.fallback_max_side <= 0:
                        metric_row = empty_metric_row(f"error:{type(error).__name__}")
                    else:
                        try:
                            resize_value = args.fallback_max_side
                            sar_fallback, sar_sx, sar_sy = resize_to_max_side(
                                sar_image, resize_value
                            )
                            optical_fallback, opt_sx, opt_sy = resize_to_max_side(
                                optical_image, resize_value
                            )
                            sar_tiepoints_scaled = sar_tiepoints_xy.copy()
                            sar_tiepoints_scaled[:, 0] *= sar_sx
                            sar_tiepoints_scaled[:, 1] *= sar_sy
                            optical_tiepoints_scaled = optical_tiepoints_xy.copy()
                            optical_tiepoints_scaled[:, 0] *= opt_sx
                            optical_tiepoints_scaled[:, 1] *= opt_sy

                            metric_row = evaluate_pair(
                                matcher=matcher,
                                mode=args.mode,
                                sar_image=sar_fallback,
                                optical_image=optical_fallback,
                                sar_tiepoints_xy=sar_tiepoints_scaled,
                                optical_tiepoints_xy=optical_tiepoints_scaled,
                                normalization=normalization,
                                tile_size=args.tile_size,
                                tile_overlap=args.tile_overlap,
                                min_tile_inliers=args.min_tile_inliers,
                                geometry_model=args.geometry_model,
                                ransac_reproj_threshold=args.ransac_reproj_threshold,
                                coverage_grid_size=args.coverage_grid_size,
                                device=args.device,
                                bf16_on_oom=args.bf16_on_oom,
                            )
                            metric_row["status"] = f"{metric_row['status']}:fallback"
                        except Exception as fallback_error:
                            metric_row = empty_metric_row(
                                f"error:{type(error).__name__}/{type(fallback_error).__name__}"
                            )

                records.append(
                    {
                        "matcher": matcher_name,
                        "normalization": normalization,
                        "mode": args.mode,
                        "case_id": pair.case_id,
                        "pair_id": pair.pair_id,
                        "tiepoints": int(sar_tiepoints_xy.shape[0]),
                        "max_side": int(resize_value),
                        "sample_size": int(args.sample_size),
                        "sample_seed": int(args.sample_seed),
                        "tile_size": int(args.tile_size),
                        "tile_overlap": int(args.tile_overlap),
                        "geometry_model": str(args.geometry_model),
                        "status_category": str(metric_row["status"]).split(":")[0],
                        **metric_row,
                    }
                )

    frame = pd.DataFrame.from_records(records)
    summary = (
        frame.groupby(["matcher", "normalization", "mode", "geometry_model"], dropna=False)
        .agg(
            pairs=("pair_id", "count"),
            ok_pairs=(
                "status",
                lambda values: int(np.sum(values.astype(str).str.startswith("ok"))),
            ),
            failure_rate=(
                "status",
                lambda values: float(np.mean(~values.astype(str).str.startswith("ok"))),
            ),
            mean_error_px=("mean_error_px", "mean"),
            median_error_px=("median_error_px", "mean"),
            success_at_1px=("success_at_1px", "mean"),
            success_at_2px=("success_at_2px", "mean"),
            success_at_3px=("success_at_3px", "mean"),
            success_at_5px=("success_at_5px", "mean"),
            success_at_10px=("success_at_10px", "mean"),
            mean_elapsed_sec=("mean_elapsed_sec", "mean"),
            mean_tiles_used=("tiles_used", "mean"),
            mean_tile_trials=("tile_trials", "mean"),
            mean_inlier_ratio=("inlier_ratio", "mean"),
            mean_spatial_coverage=("spatial_coverage", "mean"),
            bf16_pair_rate=(
                "precision_mode",
                lambda values: float(np.mean(values.astype(str).str.contains("bf16", na=False))),
            ),
        )
        .reset_index()
        .sort_values(["mean_error_px", "success_at_5px"], ascending=[False, True])
    )

    ci_records: list[dict[str, Any]] = []
    group_keys = frame[["matcher", "normalization", "mode", "geometry_model"]].drop_duplicates()
    for key in group_keys.itertuples(index=False):
        matcher = str(key.matcher)
        normalization = str(key.normalization)
        mode = str(key.mode)
        geometry_model = str(key.geometry_model)
        group = frame.loc[
            (frame["matcher"].astype(str) == matcher)
            & (frame["normalization"].astype(str) == normalization)
            & (frame["mode"].astype(str) == mode)
            & (frame["geometry_model"].astype(str) == geometry_model)
        ]
        ci_records.append(
            {
                "matcher": matcher,
                "normalization": normalization,
                "mode": mode,
                "geometry_model": geometry_model,
                **group_bootstrap_ci_row(
                    group,
                    bootstrap_samples=int(args.bootstrap_samples),
                    confidence_level=float(args.bootstrap_ci),
                    sample_seed=int(args.sample_seed),
                ),
            }
        )

    ci_frame = pd.DataFrame.from_records(ci_records)
    summary = summary.merge(
        ci_frame,
        on=["matcher", "normalization", "mode", "geometry_model"],
        how="left",
    )
    return frame, summary


def main() -> None:
    args = parse_args()
    detailed, summary = run_experiments(args)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(args.output_csv, index=False)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)

    print(summary.to_string(index=False))
    print(f"\nWrote detailed results to {args.output_csv}")
    print(f"Wrote summary results to {args.summary_csv}")


if __name__ == "__main__":
    main()
