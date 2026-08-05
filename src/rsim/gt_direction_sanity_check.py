from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import vismatch

from rsim.dataset_loaders import discover_srif_pairs
from rsim.spacenet9_matcher_benchmark import (
    parse_matcher_spec,
    run_matcher_with_precision,
    to_matcher_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--srif-dir", type=Path, default=Path("data/srif"))
    parser.add_argument(
        "--srif-subdatasets",
        nargs="+",
        default=["Optical-SAR", "Optical-Optical", "Optical-Infrared"],
    )
    parser.add_argument("--matcher", type=str, default="xfeat")
    parser.add_argument("--normalization", type=str, default="percentile")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bf16-on-oom", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-num-keypoints", type=int, default=4096)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-csv", type=Path, default=Path("outputs/gt_direction_sanity_details.csv")
    )
    parser.add_argument(
        "--summary-csv", type=Path, default=Path("outputs/gt_direction_sanity_summary.csv")
    )
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        msg = f"Failed to read image: {path}"
        raise FileNotFoundError(msg)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_homography(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape == (3, 3):
        return matrix
    if matrix.shape == (2, 3):
        homography = np.eye(3, dtype=np.float64)
        homography[:2, :3] = matrix
        return homography
    msg = f"Expected transform shape (2,3) or (3,3) in {path}, got {matrix.shape}."
    raise ValueError(msg)


def project_points(points_xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    ones = np.ones((points_xy.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points_xy.astype(np.float64), ones], axis=1)
    projected_h = (homography @ points_h.T).T
    z = projected_h[:, 2:3]
    valid = np.abs(z) > 1e-12
    projected = np.full((points_xy.shape[0], 2), np.nan, dtype=np.float64)
    projected[valid[:, 0]] = projected_h[valid[:, 0], :2] / z[valid[:, 0]]
    return projected


def direction_metrics(
    matched_src_xy: np.ndarray, matched_dst_xy: np.ndarray, homography: np.ndarray
) -> dict[str, Any]:
    if matched_src_xy.shape[0] == 0 or matched_dst_xy.shape[0] == 0:
        return {
            "status": "no_matches",
            "num_matches": 0,
            "forward_median_error": float("nan"),
            "reverse_median_error": float("nan"),
            "forward_mean_error": float("nan"),
            "reverse_mean_error": float("nan"),
            "forward_better": float("nan"),
            "best_direction": "undetermined",
            "error_ratio_reverse_over_forward": float("nan"),
        }

    try:
        homography_inv = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return {
            "status": "non_invertible_gt",
            "num_matches": int(matched_src_xy.shape[0]),
            "forward_median_error": float("nan"),
            "reverse_median_error": float("nan"),
            "forward_mean_error": float("nan"),
            "reverse_mean_error": float("nan"),
            "forward_better": float("nan"),
            "best_direction": "undetermined",
            "error_ratio_reverse_over_forward": float("nan"),
        }

    pred_forward = project_points(matched_src_xy, homography)
    pred_reverse = project_points(matched_src_xy, homography_inv)

    err_forward = np.linalg.norm(pred_forward - matched_dst_xy, axis=1)
    err_reverse = np.linalg.norm(pred_reverse - matched_dst_xy, axis=1)

    finite_mask = np.isfinite(err_forward) & np.isfinite(err_reverse)
    if not np.any(finite_mask):
        return {
            "status": "invalid_projection",
            "num_matches": int(matched_src_xy.shape[0]),
            "forward_median_error": float("nan"),
            "reverse_median_error": float("nan"),
            "forward_mean_error": float("nan"),
            "reverse_mean_error": float("nan"),
            "forward_better": float("nan"),
            "best_direction": "undetermined",
            "error_ratio_reverse_over_forward": float("nan"),
        }

    err_forward = err_forward[finite_mask]
    err_reverse = err_reverse[finite_mask]
    forward_median = float(np.median(err_forward))
    reverse_median = float(np.median(err_reverse))
    forward_mean = float(np.mean(err_forward))
    reverse_mean = float(np.mean(err_reverse))

    ratio = float(reverse_median / max(forward_median, 1e-12))
    margin = 0.95
    if forward_median < reverse_median * margin:
        best_direction = "forward"
        forward_better = 1.0
    elif reverse_median < forward_median * margin:
        best_direction = "reverse"
        forward_better = 0.0
    else:
        best_direction = "undetermined"
        forward_better = float("nan")

    return {
        "status": "ok",
        "num_matches": int(err_forward.shape[0]),
        "forward_median_error": forward_median,
        "reverse_median_error": reverse_median,
        "forward_mean_error": forward_mean,
        "reverse_mean_error": reverse_mean,
        "forward_better": forward_better,
        "best_direction": best_direction,
        "error_ratio_reverse_over_forward": ratio,
    }


def evaluate_pair_direction(
    matcher: vismatch.BaseMatcher,
    src_image: np.ndarray,
    dst_image: np.ndarray,
    homography: np.ndarray,
    normalization: str,
    device: str,
    bf16_on_oom: bool,
) -> dict[str, Any]:
    src_tensor = to_matcher_tensor(src_image, normalization, modality="sar")
    dst_tensor = to_matcher_tensor(dst_image, normalization, modality="optical")
    result, _ = run_matcher_with_precision(
        matcher=matcher,
        sar_tensor=src_tensor,
        optical_tensor=dst_tensor,
        device=device,
        bf16_on_oom=bf16_on_oom,
    )
    matched_src = np.asarray(result.get("matched_kpts0", np.empty((0, 2))), dtype=np.float64)
    matched_dst = np.asarray(result.get("matched_kpts1", np.empty((0, 2))), dtype=np.float64)
    return direction_metrics(matched_src, matched_dst, homography)


def sample_items(items: list[Any], sample_size: int, seed: int) -> list[Any]:
    if sample_size <= 0 or sample_size >= len(items):
        return items
    rng = np.random.default_rng(seed)
    selected_idx = sorted(rng.choice(len(items), size=sample_size, replace=False).tolist())
    return [items[index] for index in selected_idx]


def main() -> None:
    args = parse_args()
    matcher = vismatch.get_matcher(
        matcher_name=parse_matcher_spec(args.matcher),
        device=args.device,
        max_num_keypoints=args.max_num_keypoints,
    )

    srif_pairs = discover_srif_pairs(args.srif_dir, args.srif_subdatasets)
    srif_eval_pairs = sample_items(srif_pairs, int(args.sample_size), int(args.seed))

    image_cache: dict[Path, np.ndarray] = {}
    transform_cache: dict[Path, np.ndarray] = {}
    records: list[dict[str, Any]] = []

    for pair in srif_eval_pairs:
        if pair.image1_path not in image_cache:
            image_cache[pair.image1_path] = load_rgb(pair.image1_path)
        if pair.image2_path not in image_cache:
            image_cache[pair.image2_path] = load_rgb(pair.image2_path)
        if pair.transform_path not in transform_cache:
            transform_cache[pair.transform_path] = load_homography(pair.transform_path)

        metrics = evaluate_pair_direction(
            matcher=matcher,
            src_image=image_cache[pair.image1_path],
            dst_image=image_cache[pair.image2_path],
            homography=transform_cache[pair.transform_path],
            normalization=args.normalization,
            device=args.device,
            bf16_on_oom=args.bf16_on_oom,
        )
        records.append(
            {
                "dataset": "srif",
                "case_id": pair.case_id,
                "pair_id": pair.pair_id,
                "src_path": str(pair.image1_path),
                "dst_path": str(pair.image2_path),
                "transform_path": str(pair.transform_path),
                **metrics,
            }
        )

    details = pd.DataFrame.from_records(records)
    if details.empty:
        msg = "No records produced for GT direction sanity check."
        raise RuntimeError(msg)

    summary_records: list[dict[str, Any]] = []
    for dataset, group in details.groupby("dataset"):
        ok = group[group["status"] == "ok"]
        summary_records.append(
            {
                "dataset": dataset,
                "pairs": int(group.shape[0]),
                "ok_pairs": int(ok.shape[0]),
                "forward_votes": int((ok["best_direction"] == "forward").sum()),
                "reverse_votes": int((ok["best_direction"] == "reverse").sum()),
                "undetermined_votes": int((ok["best_direction"] == "undetermined").sum()),
                "mean_forward_median_error": float(ok["forward_median_error"].mean())
                if not ok.empty
                else float("nan"),
                "mean_reverse_median_error": float(ok["reverse_median_error"].mean())
                if not ok.empty
                else float("nan"),
                "mean_error_ratio_reverse_over_forward": float(
                    ok["error_ratio_reverse_over_forward"].mean()
                )
                if not ok.empty
                else float("nan"),
            }
        )
    summary = pd.DataFrame.from_records(summary_records)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.output_csv, index=False)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)

    print(summary.to_string(index=False))
    print(f"\nWrote GT direction details to {args.output_csv}")
    print(f"Wrote GT direction summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
