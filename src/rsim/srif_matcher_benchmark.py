from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import cv2
import numpy as np
import pandas as pd
import vismatch

from rsim.dataset_loaders import discover_srif_pairs
from rsim.matcher_roster import DEFAULT_MATCHERS
from rsim.spacenet9_matcher_benchmark import (
    empty_metric_row,
    evaluate_pair,
    group_bootstrap_ci_row,
    parse_matcher_spec,
    resize_to_max_side,
    sample_pairs,
)
from rsim.spacenet9_matcher_benchmark import (
    read_image as read_raster_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/srif"))
    parser.add_argument(
        "--subdatasets",
        nargs="+",
        default=["Optical-SAR", "Optical-Optical", "Optical-Infrared"],
        help="SRIF subdatasets to evaluate.",
    )
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
        default="affine",
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
        "--srif-grid-size",
        type=int,
        default=20,
        help="Grid size used to synthesize SRIF tiepoints from affine labels.",
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
        default=Path("outputs/srif_matcher_results.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/srif_matcher_summary.csv"),
    )
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is not None:
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return read_raster_image(path)


def load_affine_transform(transform_path: Path) -> np.ndarray:
    matrix = np.loadtxt(transform_path, dtype=np.float64)
    if matrix.shape != (2, 3):
        msg = f"Expected 2x3 affine transform in {transform_path}, got {matrix.shape}."
        raise ValueError(msg)

    transform = np.eye(3, dtype=np.float64)
    transform[:2, :3] = matrix
    return transform


def synthesize_tiepoints(
    transform_1_to_2: np.ndarray,
    image1_shape: tuple[int, int],
    image2_shape: tuple[int, int],
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    height1, width1 = image1_shape
    height2, width2 = image2_shape

    grid_size = max(4, int(grid_size))
    xs = np.linspace(0.0, float(width1 - 1), num=grid_size, dtype=np.float64)
    ys = np.linspace(0.0, float(height1 - 1), num=grid_size, dtype=np.float64)
    mesh_x, mesh_y = np.meshgrid(xs, ys)

    image1_xy = np.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], axis=1)
    image1_h = np.concatenate(
        [image1_xy, np.ones((image1_xy.shape[0], 1), dtype=np.float64)], axis=1
    )
    image2_h = (transform_1_to_2 @ image1_h.T).T
    image2_xy = image2_h[:, :2]

    valid = (
        (image2_xy[:, 0] >= 0.0)
        & (image2_xy[:, 0] <= float(width2 - 1))
        & (image2_xy[:, 1] >= 0.0)
        & (image2_xy[:, 1] <= float(height2 - 1))
    )

    image1_valid = image1_xy[valid]
    image2_valid = image2_xy[valid]
    if image1_valid.shape[0] < 8:
        msg = "Insufficient valid tiepoints generated from SRIF transform."
        raise ValueError(msg)
    return image1_valid, image2_valid


def run_experiments(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    vismatch.disable_progress_bars()
    discovered_pairs = discover_srif_pairs(args.data_dir, args.subdatasets)
    pairs = sample_pairs(
        discovered_pairs,
        sample_size=int(args.sample_size),
        sample_seed=int(args.sample_seed),
        sample_by_case=bool(args.sample_by_case),
    )

    records: list[dict[str, Any]] = []
    image_cache: dict[Path, np.ndarray] = {}
    transform_cache: dict[Path, np.ndarray] = {}

    for matcher_name in args.matchers:
        matcher_spec = parse_matcher_spec(matcher_name)
        try:
            matcher = vismatch.get_matcher(
                matcher_name=matcher_spec,
                device=args.device,
                max_num_keypoints=args.max_num_keypoints,
            )
        except Exception as init_error:
            records.extend(
                {
                    "matcher": matcher_name,
                    "normalization": normalization,
                    "mode": args.mode,
                    "geometry_model": str(args.geometry_model),
                    "subdataset": pair.subdataset,
                    "case_id": pair.case_id,
                    "pair_id": pair.pair_id,
                    "tiepoints": np.nan,
                    "max_side": int(args.max_side),
                    "sample_size": int(args.sample_size),
                    "sample_seed": int(args.sample_seed),
                    "tile_size": int(args.tile_size),
                    "tile_overlap": int(args.tile_overlap),
                    "status_category": "init_error",
                    **empty_metric_row(f"init_error:{type(init_error).__name__}"),
                }
                for normalization in args.normalizations
                for pair in pairs
            )
            continue

        for normalization in args.normalizations:
            for pair in pairs:
                if pair.image1_path not in image_cache:
                    image_cache[pair.image1_path] = read_image(pair.image1_path)
                if pair.image2_path not in image_cache:
                    image_cache[pair.image2_path] = read_image(pair.image2_path)
                if pair.transform_path not in transform_cache:
                    transform_cache[pair.transform_path] = load_affine_transform(
                        pair.transform_path
                    )

                image1 = image_cache[pair.image1_path]
                image2 = image_cache[pair.image2_path]
                transform_1_to_2 = transform_cache[pair.transform_path]
                image1_tiepoints_xy, image2_tiepoints_xy = synthesize_tiepoints(
                    transform_1_to_2,
                    image1_shape=image1.shape[:2],
                    image2_shape=image2.shape[:2],
                    grid_size=int(args.srif_grid_size),
                )

                resize_value = args.max_side
                image1_resized, image1_sx, image1_sy = resize_to_max_side(image1, resize_value)
                image2_resized, image2_sx, image2_sy = resize_to_max_side(image2, resize_value)

                image1_tiepoints_scaled = image1_tiepoints_xy.copy()
                image1_tiepoints_scaled[:, 0] *= image1_sx
                image1_tiepoints_scaled[:, 1] *= image1_sy
                image2_tiepoints_scaled = image2_tiepoints_xy.copy()
                image2_tiepoints_scaled[:, 0] *= image2_sx
                image2_tiepoints_scaled[:, 1] *= image2_sy

                try:
                    metric_row = evaluate_pair(
                        matcher=matcher,
                        mode=args.mode,
                        sar_image=image1_resized,
                        optical_image=image2_resized,
                        sar_tiepoints_xy=image1_tiepoints_scaled,
                        optical_tiepoints_xy=image2_tiepoints_scaled,
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
                            image1_fallback, image1_sx, image1_sy = resize_to_max_side(
                                image1, resize_value
                            )
                            image2_fallback, image2_sx, image2_sy = resize_to_max_side(
                                image2, resize_value
                            )

                            image1_tiepoints_scaled = image1_tiepoints_xy.copy()
                            image1_tiepoints_scaled[:, 0] *= image1_sx
                            image1_tiepoints_scaled[:, 1] *= image1_sy
                            image2_tiepoints_scaled = image2_tiepoints_xy.copy()
                            image2_tiepoints_scaled[:, 0] *= image2_sx
                            image2_tiepoints_scaled[:, 1] *= image2_sy

                            metric_row = evaluate_pair(
                                matcher=matcher,
                                mode=args.mode,
                                sar_image=image1_fallback,
                                optical_image=image2_fallback,
                                sar_tiepoints_xy=image1_tiepoints_scaled,
                                optical_tiepoints_xy=image2_tiepoints_scaled,
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
                        "geometry_model": str(args.geometry_model),
                        "subdataset": pair.subdataset,
                        "case_id": pair.case_id,
                        "pair_id": pair.pair_id,
                        "tiepoints": int(image1_tiepoints_xy.shape[0]),
                        "max_side": int(resize_value),
                        "sample_size": int(args.sample_size),
                        "sample_seed": int(args.sample_seed),
                        "tile_size": int(args.tile_size),
                        "tile_overlap": int(args.tile_overlap),
                        "status_category": str(metric_row["status"]).split(":")[0],
                        **metric_row,
                    }
                )

    frame = pd.DataFrame.from_records(records)
    summary = (
        frame.groupby(
            ["subdataset", "matcher", "normalization", "mode", "geometry_model"], dropna=False
        )
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
        .sort_values(
            ["subdataset", "mean_error_px", "success_at_5px"],
            ascending=[True, True, False],
        )
    )

    ci_records: list[dict[str, Any]] = []
    group_keys = frame[
        ["subdataset", "matcher", "normalization", "mode", "geometry_model"]
    ].drop_duplicates()
    for key in group_keys.itertuples(index=False):
        subdataset = str(key.subdataset)
        matcher = str(key.matcher)
        normalization = str(key.normalization)
        mode = str(key.mode)
        geometry_model = str(key.geometry_model)

        group = frame.loc[
            (frame["subdataset"].astype(str) == subdataset)
            & (frame["matcher"].astype(str) == matcher)
            & (frame["normalization"].astype(str) == normalization)
            & (frame["mode"].astype(str) == mode)
            & (frame["geometry_model"].astype(str) == geometry_model)
        ]

        ci_records.append(
            {
                "subdataset": subdataset,
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
        on=["subdataset", "matcher", "normalization", "mode", "geometry_model"],
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
