"""Re-run the 16-config protocol sweep with top-performing matchers.

The original protocol-sensitivity table averages over LoFTR/XFeat. This script
runs the same protocol grid (geometry x max_side x overlap x min_inliers) with
the top-performing matchers to confirm protocol effects generalize.
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
from pathlib import Path

import pandas as pd
import torch
import vismatch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "spacenet9" / "train"
OUT_DIR = ROOT / "outputs" / "protocol_sweep_top_matchers"
RUN_DIR = OUT_DIR / "runs"

_MINIMA_ROOT = Path(vismatch.__file__).parent / "third_party" / "MINIMA"
if _MINIMA_ROOT.is_dir():
    _pp = os.environ.get("PYTHONPATH", "")
    if str(_MINIMA_ROOT) not in _pp:
        os.environ["PYTHONPATH"] = str(_MINIMA_ROOT) + (os.pathsep + _pp if _pp else "")

# Protocol grid matching Table 6
GEOMETRY_MODELS = ["affine", "homography"]
MAX_SIDES = [1024, 2048]
TILE_OVERLAPS = [128, 256]
MIN_TILE_INLIERS = [4, 8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Protocol sweep with top matchers (geometry x max_side x overlap x min_inliers)"
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--matchers",
        nargs="+",
        default=["roma", "xoftr", "matchanything-eloftr", "minima-roma"],
    )
    parser.add_argument("--normalizations", nargs="+", default=["percentile"])
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=512)
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


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    configs = list(itertools.product(GEOMETRY_MODELS, MAX_SIDES, TILE_OVERLAPS, MIN_TILE_INLIERS))
    print(f"Protocol sweep: {len(configs)} configs x {len(args.matchers)} matchers")

    summary_rows: list[pd.DataFrame] = []
    detail_rows: list[pd.DataFrame] = []

    for geom, max_side, overlap, min_inliers in tqdm(configs, desc="protocol sweep"):
        tag = f"{geom}_ms{max_side}_ov{overlap}_mi{min_inliers}"
        details = RUN_DIR / f"{tag}_details.csv"
        summary = RUN_DIR / f"{tag}_summary.csv"

        extra = [
            "--matchers",
            *args.matchers,
            "--normalizations",
            *args.normalizations,
            "--device",
            device,
            "--bf16-on-oom",
            "--max-side",
            str(max_side),
            "--tile-size",
            str(args.tile_size),
            "--tile-overlap",
            str(overlap),
            "--geometry-model",
            geom,
            "--min-tile-inliers",
            str(min_inliers),
            "--ransac-reproj-threshold",
            "3.0",
        ]
        if args.sample_size > 0:
            extra.extend(["--sample-size", str(args.sample_size)])

        _run_benchmark(extra, details, summary)

        sf = pd.read_csv(summary)
        sf["config_tag"] = tag
        summary_rows.append(sf)

        df = pd.read_csv(details)
        df["config_tag"] = tag
        detail_rows.append(df)

    all_summary = pd.concat(summary_rows, ignore_index=True)
    all_details = pd.concat(detail_rows, ignore_index=True)
    all_summary.to_csv(OUT_DIR / "protocol_sweep_summary.csv", index=False)
    all_details.to_csv(OUT_DIR / "protocol_sweep_details.csv", index=False)
    print(f"Wrote {len(all_summary)} summary rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
