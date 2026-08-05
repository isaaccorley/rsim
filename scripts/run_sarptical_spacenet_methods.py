from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from rsim.matcher_roster import SARPTICAL_BASELINE_MATCHERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods-csv",
        type=Path,
        default=Path("outputs/final_master_leaderboard_best_per_matcher.csv"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/sarptical/patch_SAR_OPT_SQUARE"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-queries", type=int, default=40)
    parser.add_argument("--negatives-per-query", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sarptical_grid"))
    parser.add_argument(
        "--consolidated-summary",
        type=Path,
        default=Path("outputs/sarptical_grid_all_methods_summary.csv"),
    )
    parser.add_argument(
        "--include-baselines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include additional baseline methods (Kornia-style classical/feature baselines).",
    )
    parser.add_argument(
        "--baseline-matchers",
        nargs="+",
        default=SARPTICAL_BASELINE_MATCHERS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = pd.read_csv(args.methods_csv)
    methods = (
        methods[["matcher", "normalization"]]
        .drop_duplicates()
        .sort_values(["matcher", "normalization"])
    )

    if args.include_baselines:
        baseline_rows = pd.DataFrame(
            {
                "matcher": list(args.baseline_matchers),
                "normalization": ["identity"] * len(args.baseline_matchers),
            }
        )
        methods = pd.concat([methods, baseline_rows], ignore_index=True)
        methods = methods.drop_duplicates(subset=["matcher", "normalization"]).sort_values(
            ["matcher", "normalization"]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, str | int | float]] = []
    summary_frames: list[pd.DataFrame] = []

    method_rows = list(methods.itertuples(index=False))
    progress = tqdm(method_rows, desc="sarptical methods", unit="method")

    for row in progress:
        matcher = str(row.matcher)
        normalization = str(row.normalization)
        progress.set_postfix(matcher=matcher, norm=normalization)
        details_path = args.output_dir / f"{matcher}_{normalization}_details.csv"
        summary_path = args.output_dir / f"{matcher}_{normalization}_summary.csv"

        command = [
            "uv",
            "run",
            "python",
            "-m",
            "rsim.sarptical_pair_eval",
            "--data-dir",
            str(args.data_dir),
            "--matcher",
            matcher,
            "--normalization",
            normalization,
            "--device",
            args.device,
            "--max-queries",
            str(args.max_queries),
            "--negatives-per-query",
            str(args.negatives_per_query),
            "--seed",
            str(args.seed),
            "--output-csv",
            str(details_path),
            "--summary-csv",
            str(summary_path),
        ]

        print(
            f"\n=== Running SARptical: matcher={matcher} normalization={normalization} ===",
            flush=True,
        )
        completed = subprocess.run(command, check=False)

        status = "ok" if completed.returncode == 0 else f"failed:{completed.returncode}"
        progress.set_postfix(matcher=matcher, norm=normalization, status=status)
        run_rows.append(
            {
                "matcher": matcher,
                "normalization": normalization,
                "status": status,
                "summary_csv": str(summary_path),
            }
        )

        if completed.returncode == 0 and summary_path.exists():
            summary_frame = pd.read_csv(summary_path)
            summary_frame["run_status"] = status
            summary_frames.append(summary_frame)

    run_status_frame = pd.DataFrame.from_records(run_rows)
    run_status_path = args.output_dir / "run_status.csv"
    run_status_frame.to_csv(run_status_path, index=False)

    if summary_frames:
        consolidated = pd.concat(summary_frames, ignore_index=True)
    else:
        consolidated = pd.DataFrame(columns=["matcher", "normalization", "run_status"])
    args.consolidated_summary.parent.mkdir(parents=True, exist_ok=True)
    consolidated.to_csv(args.consolidated_summary, index=False)

    print("\nCompleted SARptical all-method sweep.")
    print(f"Run status: {run_status_path}")
    print(f"Consolidated summary: {args.consolidated_summary}")

    failed_count = int((run_status_frame["status"].astype(str) != "ok").sum())
    if failed_count > 0:
        print(f"Failed runs: {failed_count}")
        sys.exit(1)


if __name__ == "__main__":
    main()
