from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import vismatch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]

# Workaround: vismatch's MINIMA load_model.py uses relative imports from its
# repo root.  Add the vendored MINIMA directory to PYTHONPATH so subprocesses
# can resolve ``from src.utils.load_model import ...``.
_MINIMA_ROOT = Path(vismatch.__file__).parent / "third_party" / "MINIMA"
if _MINIMA_ROOT.is_dir():
    _pp = os.environ.get("PYTHONPATH", "")
    if str(_MINIMA_ROOT) not in _pp:
        os.environ["PYTHONPATH"] = str(_MINIMA_ROOT) + (os.pathsep + _pp if _pp else "")
DATA_DIR = ROOT / "data" / "spacenet9" / "train"
OUT_DIR = ROOT / "outputs" / "extended_ablations"
RUN_DIR = OUT_DIR / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--matchers",
        nargs="+",
        default=[
            "minima-roma",
            "roma",
            "loftr",
            "xfeat",
            "romav2",
            "tiny-roma",
        ],
    )
    parser.add_argument("--normalizations", nargs="+", default=["percentile"])
    parser.add_argument("--base-max-side", type=int, default=1024)
    parser.add_argument("--base-tile-size", type=int, default=512)
    parser.add_argument("--base-tile-overlap", type=int, default=256)
    parser.add_argument(
        "--base-geometry", type=str, default="affine", choices=["affine", "homography"]
    )
    parser.add_argument("--sample-size", type=int, default=0)
    return parser.parse_args()


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _run_benchmark(extra_args: list[str], details_csv: Path, summary_csv: Path) -> None:
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "rsim.spacenet9_matcher_benchmark",
        "--data-dir",
        str(DATA_DIR),
        "--mode",
        "tiled",
        "--output-csv",
        str(details_csv),
        "--summary-csv",
        str(summary_csv),
        *extra_args,
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def run_threshold_robustness(base_args: list[str]) -> None:
    thresholds = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
    summary_rows: list[pd.DataFrame] = []
    detail_rows: list[pd.DataFrame] = []
    for threshold in tqdm(thresholds, desc="threshold sweep", unit="setting"):
        tag = str(threshold).replace(".", "p")
        details = RUN_DIR / f"threshold_t{tag}_details.csv"
        summary = RUN_DIR / f"threshold_t{tag}_summary.csv"
        extra = [*base_args, "--ransac-reproj-threshold", str(threshold)]
        _run_benchmark(extra, details, summary)
        summary_frame = pd.read_csv(summary)
        detail_frame = pd.read_csv(details)
        summary_frame["ransac_reproj_threshold"] = threshold
        detail_frame["ransac_reproj_threshold"] = threshold
        summary_rows.append(summary_frame)
        detail_rows.append(detail_frame)

    summary = pd.concat(summary_rows, ignore_index=True)
    details = pd.concat(detail_rows, ignore_index=True)
    summary["ablation"] = "ransac_threshold"
    details["ablation"] = "ransac_threshold"
    summary.to_csv(OUT_DIR / "threshold_robustness_summary.csv", index=False)
    details.to_csv(OUT_DIR / "threshold_robustness_details.csv", index=False)


def run_inlier_gating(base_args: list[str]) -> None:
    min_inliers_list = [2, 4, 6, 8, 10]
    summary_rows: list[pd.DataFrame] = []
    detail_rows: list[pd.DataFrame] = []
    for min_inliers in tqdm(min_inliers_list, desc="inlier gating", unit="setting"):
        details = RUN_DIR / f"inliers_i{min_inliers}_details.csv"
        summary = RUN_DIR / f"inliers_i{min_inliers}_summary.csv"
        extra = [
            *base_args,
            "--min-tile-inliers",
            str(min_inliers),
            "--ransac-reproj-threshold",
            "3.0",
        ]
        _run_benchmark(extra, details, summary)
        summary_frame = pd.read_csv(summary)
        detail_frame = pd.read_csv(details)
        summary_frame["min_tile_inliers"] = min_inliers
        detail_frame["min_tile_inliers"] = min_inliers
        summary_rows.append(summary_frame)
        detail_rows.append(detail_frame)

    summary = pd.concat(summary_rows, ignore_index=True)
    details = pd.concat(detail_rows, ignore_index=True)
    summary["ablation"] = "min_tile_inliers"
    details["ablation"] = "min_tile_inliers"
    summary.to_csv(OUT_DIR / "inlier_gating_summary.csv", index=False)
    details.to_csv(OUT_DIR / "inlier_gating_details.csv", index=False)


def run_keypoint_budget(base_args: list[str]) -> None:
    budgets = [512, 1024, 2048, 4096, 8192]
    summary_rows: list[pd.DataFrame] = []
    detail_rows: list[pd.DataFrame] = []
    for budget in tqdm(budgets, desc="keypoint budget", unit="setting"):
        details = RUN_DIR / f"keypoints_k{budget}_details.csv"
        summary = RUN_DIR / f"keypoints_k{budget}_summary.csv"
        extra = [*base_args, "--max-num-keypoints", str(budget), "--ransac-reproj-threshold", "3.0"]
        _run_benchmark(extra, details, summary)
        summary_frame = pd.read_csv(summary)
        detail_frame = pd.read_csv(details)
        summary_frame["max_num_keypoints"] = budget
        detail_frame["max_num_keypoints"] = budget
        summary_rows.append(summary_frame)
        detail_rows.append(detail_frame)

    summary = pd.concat(summary_rows, ignore_index=True)
    details = pd.concat(detail_rows, ignore_index=True)
    summary["ablation"] = "max_num_keypoints"
    details["ablation"] = "max_num_keypoints"
    summary.to_csv(OUT_DIR / "keypoint_budget_summary.csv", index=False)
    details.to_csv(OUT_DIR / "keypoint_budget_details.csv", index=False)


def run_retrieval_conditioned_analysis() -> None:
    details_path = ROOT / "outputs" / "sarptical_phase5_details.csv"
    if not details_path.exists():
        raise FileNotFoundError(f"Missing SARptical details: {details_path}")

    df = pd.read_csv(details_path)
    shortlist_sizes = [5, 10, 20, 40]
    out_rows: list[dict[str, float | int | str]] = []

    matcher_names = sorted(df["matcher"].astype(str).unique())
    normalization_names = sorted(df["normalization"].astype(str).unique())

    for matcher in tqdm(matcher_names, desc="retrieval analysis", unit="matcher"):
        for normalization in normalization_names:
            group = df[(df["matcher"] == matcher) & (df["normalization"] == normalization)]
            if group.empty:
                continue
            for k in shortlist_sizes:
                hit_at_1 = []
                hit_at_k = []
                pos_score_gaps = []
                for _, q in group.groupby("query_point_id"):
                    ranked = q.sort_values("score", ascending=False)
                    topk = ranked.head(k)
                    labels_topk = topk["label"].to_numpy(dtype=int)
                    hit_at_k.append(float(np.any(labels_topk == 1)))
                    hit_at_1.append(float(labels_topk[0] == 1))

                    pos = ranked[ranked["label"] == 1]
                    neg = ranked[ranked["label"] == 0]
                    if not pos.empty and not neg.empty:
                        pos_score_gaps.append(float(pos["score"].max() - neg["score"].max()))

                out_rows.append(
                    {
                        "matcher": matcher,
                        "normalization": normalization,
                        "shortlist_k": int(k),
                        "queries": int(group["query_point_id"].nunique()),
                        "recall_at_1": float(np.mean(hit_at_1)) if hit_at_1 else np.nan,
                        "recall_at_k": float(np.mean(hit_at_k)) if hit_at_k else np.nan,
                        "mean_pos_neg_score_gap": float(np.mean(pos_score_gaps))
                        if pos_score_gaps
                        else np.nan,
                    }
                )

    result = pd.DataFrame(out_rows).sort_values(["matcher", "normalization", "shortlist_k"])
    result.to_csv(OUT_DIR / "sarptical_shortlist_conditioned.csv", index=False)


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    base_args = [
        "--matchers",
        *args.matchers,
        "--normalizations",
        *args.normalizations,
        "--device",
        device,
        "--bf16-on-oom",
        "--max-side",
        str(args.base_max_side),
        "--tile-size",
        str(args.base_tile_size),
        "--tile-overlap",
        str(args.base_tile_overlap),
        "--geometry-model",
        args.base_geometry,
    ]
    if args.sample_size > 0:
        base_args.extend(["--sample-size", str(args.sample_size)])

    run_threshold_robustness(base_args)
    run_inlier_gating(base_args)
    run_keypoint_budget(base_args)
    run_retrieval_conditioned_analysis()

    print(f"Wrote extended ablation outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
