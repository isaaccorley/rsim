from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from rsim.spacenet9_matcher_benchmark import run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/spacenet9/train"))
    parser.add_argument("--matchers", nargs="+", default=["xfeat", "loftr", "roma"])
    parser.add_argument("--normalizations", nargs="+", default=["identity", "percentile", "zscore"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=str, choices=["full", "tiled"], default="tiled")
    parser.add_argument("--bf16-on-oom", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-num-keypoints", type=int, default=4096)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--sample-by-case", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-ci", type=float, default=95.0)
    parser.add_argument("--fallback-max-side", type=int, default=0)

    parser.add_argument("--tile-sizes", nargs="+", type=int, default=[512, 768, 1024])
    parser.add_argument("--tile-overlaps", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--min-tile-inliers-list", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--max-sides", nargs="+", type=int, default=[1536, 2048])
    parser.add_argument("--geometry-models", nargs="+", default=["homography", "affine"])
    parser.add_argument("--ransac-thresholds", nargs="+", type=float, default=[2.0, 3.0, 5.0])

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sweeps"))
    parser.add_argument("--run-name", type=str, default="spacenet9_protocol_sweep")
    return parser.parse_args()


def build_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    sweep_id = 0
    for max_side in args.max_sides:
        for geometry_model in args.geometry_models:
            for ransac_threshold in args.ransac_thresholds:
                for tile_size in args.tile_sizes:
                    for tile_overlap in args.tile_overlaps:
                        for min_tile_inliers in args.min_tile_inliers_list:
                            sweep_id += 1
                            configs.append(
                                {
                                    "sweep_id": sweep_id,
                                    "max_side": int(max_side),
                                    "geometry_model": str(geometry_model),
                                    "ransac_reproj_threshold": float(ransac_threshold),
                                    "tile_size": int(tile_size),
                                    "tile_overlap": int(tile_overlap),
                                    "min_tile_inliers": int(min_tile_inliers),
                                }
                            )
    return configs


def namespace_for_config(
    base_args: argparse.Namespace, config: dict[str, Any]
) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=base_args.data_dir,
        matchers=base_args.matchers,
        device=base_args.device,
        bf16_on_oom=base_args.bf16_on_oom,
        max_num_keypoints=base_args.max_num_keypoints,
        mode=base_args.mode,
        normalizations=base_args.normalizations,
        max_side=config["max_side"],
        fallback_max_side=base_args.fallback_max_side,
        tile_size=config["tile_size"],
        tile_overlap=config["tile_overlap"],
        min_tile_inliers=config["min_tile_inliers"],
        geometry_model=config["geometry_model"],
        ransac_reproj_threshold=config["ransac_reproj_threshold"],
        coverage_grid_size=4,
        sample_size=base_args.sample_size,
        sample_seed=base_args.sample_seed,
        sample_by_case=base_args.sample_by_case,
        bootstrap_samples=base_args.bootstrap_samples,
        bootstrap_ci=base_args.bootstrap_ci,
        output_csv=Path("unused_details.csv"),
        summary_csv=Path("unused_summary.csv"),
    )


def main() -> None:
    args = parse_args()
    configs = build_configs(args)

    all_details: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []

    for config in configs:
        cfg_ns = namespace_for_config(args, config)
        details, summary = run_experiments(cfg_ns)
        for key, value in config.items():
            details[key] = value
            summary[key] = value
        all_details.append(details)
        all_summary.append(summary)

    details_frame = pd.concat(all_details, ignore_index=True)
    summary_frame = pd.concat(all_summary, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / f"{args.run_name}_details.csv"
    summary_path = args.output_dir / f"{args.run_name}_summary.csv"

    details_frame.to_csv(details_path, index=False)
    summary_frame.to_csv(summary_path, index=False)

    print(summary_frame.to_string(index=False))
    print(f"\nWrote sweep details to {details_path}")
    print(f"Wrote sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
