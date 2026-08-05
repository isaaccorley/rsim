from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import cv2
import kornia.feature as KF
import numpy as np
import pandas as pd
import scipy.io
import vismatch

from rsim.spacenet9_matcher_benchmark import (
    fit_geometric_model,
    parse_matcher_spec,
    run_matcher_with_precision,
    to_matcher_tensor,
)


@dataclass(frozen=True)
class SarpticalQuery:
    point_id: int
    sar_path: Path
    positive_optical_paths: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/sarptical/patch_SAR_OPT_SQUARE")
    )
    parser.add_argument("--matcher", type=str, default="xfeat")
    parser.add_argument("--normalization", type=str, default="percentile")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bf16-on-oom", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-num-keypoints", type=int, default=4096)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--negatives-per-query", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-csv", type=Path, default=Path("outputs/sarptical_pair_eval_details.csv")
    )
    parser.add_argument(
        "--summary-csv", type=Path, default=Path("outputs/sarptical_pair_eval_summary.csv")
    )
    return parser.parse_args()


def discover_queries(data_dir: Path) -> list[SarpticalQuery]:
    sar_pattern = re.compile(r"^point_(\d+)_ampPatch\.mat$")
    optical_pattern = re.compile(r"^point_(\d+)_.*\.png$")

    sar_by_id: dict[int, Path] = {}
    optical_by_id: dict[int, list[Path]] = {}

    for path in sorted(data_dir.iterdir()):
        name = path.name
        sar_match = sar_pattern.match(name)
        if sar_match is not None:
            point_id = int(sar_match.group(1))
            sar_by_id[point_id] = path
            continue

        optical_match = optical_pattern.match(name)
        if optical_match is not None:
            point_id = int(optical_match.group(1))
            optical_by_id.setdefault(point_id, []).append(path)

    queries: list[SarpticalQuery] = []
    for point_id, sar_path in sar_by_id.items():
        positives = optical_by_id.get(point_id, [])
        if positives:
            queries.append(
                SarpticalQuery(
                    point_id=point_id, sar_path=sar_path, positive_optical_paths=positives
                )
            )

    if not queries:
        msg = f"No valid SARptical query pairs discovered under {data_dir}"
        raise FileNotFoundError(msg)
    return queries


def load_sar_mat_as_rgb(path: Path) -> np.ndarray:
    mat = scipy.io.loadmat(path)
    array_2d: np.ndarray | None = None
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and value.ndim == 2:
            array_2d = value.astype(np.float32)
            break
    if array_2d is None:
        msg = f"No 2D SAR matrix found in {path}"
        raise ValueError(msg)

    finite = np.isfinite(array_2d)
    if not np.any(finite):
        normalized = np.zeros_like(array_2d, dtype=np.uint8)
    else:
        valid_values = array_2d[finite]
        low = float(np.percentile(valid_values, 2.0))
        high = float(np.percentile(valid_values, 98.0))
        if high <= low:
            normalized = np.zeros_like(array_2d, dtype=np.uint8)
        else:
            scaled = np.clip((array_2d - low) / (high - low), 0.0, 1.0)
            normalized = np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)

    return np.repeat(normalized[:, :, None], 3, axis=2)


def load_optical_png_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        msg = f"Failed to read optical patch: {path}"
        raise FileNotFoundError(msg)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def is_kornia_fallback_matcher(matcher_name: str) -> bool:
    lowered = matcher_name.lower()
    if lowered.startswith("kornia:"):
        return True
    return lowered in {
        "hardnet-lightglue",
        "sift",
        "disk",
    }


def resolve_kornia_matcher_name(matcher_name: str) -> str:
    lowered = matcher_name.lower()
    if lowered.startswith("kornia:"):
        return lowered.split(":", maxsplit=1)[1]
    if lowered == "hardnet-lightglue":
        return "doghardnet-lightglue"
    if lowered == "sift":
        return "sift-snn"
    if lowered == "disk":
        return "disk-lightglue"
    return lowered


def make_torch_tensor(
    image_rgb: np.ndarray, normalization: str, modality: str, device: str
) -> torch.Tensor:
    chw = to_matcher_tensor(image_rgb, normalization, modality=modality)
    tensor = torch.from_numpy(chw).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def build_kornia_components(
    matcher_name: str, device: str, max_num_keypoints: int
) -> dict[str, Any]:
    resolved = resolve_kornia_matcher_name(matcher_name)
    torch_device = torch.device(device)
    if resolved == "sift-snn":
        sift = (
            KF.SIFTFeature(num_features=max_num_keypoints, device=torch_device)
            .to(torch_device)
            .eval()
        )
        return {"kind": resolved, "sift": sift, "device": device}
    if resolved == "sift-lightglue":
        sift = (
            KF.SIFTFeature(num_features=max_num_keypoints, device=torch_device)
            .to(torch_device)
            .eval()
        )
        matcher = KF.LightGlueMatcher("sift").to(torch_device).eval()
        return {"kind": resolved, "sift": sift, "lightglue": matcher, "device": device}
    if resolved == "disk-lightglue":
        disk = KF.DISK.from_pretrained("depth", device=torch_device).to(torch_device).eval()
        matcher = KF.LightGlueMatcher("disk").to(torch_device).eval()
        return {
            "kind": resolved,
            "disk": disk,
            "lightglue": matcher,
            "device": device,
            "max_num_keypoints": max_num_keypoints,
        }
    if resolved == "doghardnet-lightglue":
        hardnet = (
            KF.HesAffNetHardNet(num_features=max_num_keypoints, device=torch_device)
            .to(torch_device)
            .eval()
        )
        matcher = KF.LightGlueMatcher("doghardnet").to(torch_device).eval()
        return {"kind": resolved, "hardnet": hardnet, "lightglue": matcher, "device": device}

    msg = f"Unsupported Kornia matcher fallback: {matcher_name}"
    raise ValueError(msg)


def kornia_pair_score(
    components: dict[str, Any],
    sar_rgb: np.ndarray,
    optical_rgb: np.ndarray,
    normalization: str,
) -> tuple[float, int, int]:
    device = str(components["device"])
    kind = str(components["kind"])

    sar_tensor = make_torch_tensor(sar_rgb, normalization, modality="sar", device=device)
    optical_tensor = make_torch_tensor(
        optical_rgb, normalization, modality="optical", device=device
    )

    with torch.inference_mode():
        if kind in {"sift-snn", "sift-lightglue", "doghardnet-lightglue"}:
            gray0 = sar_tensor.mean(dim=1, keepdim=True)
            gray1 = optical_tensor.mean(dim=1, keepdim=True)

        if kind == "sift-snn":
            sift: KF.SIFTFeature = components["sift"]
            lafs0, _, desc0 = sift(gray0)
            lafs1, _, desc1 = sift(gray1)
            _, matches = KF.match_snn(desc0[0], desc1[0], th=0.8)
            centers0 = KF.get_laf_center(lafs0)[0]
            centers1 = KF.get_laf_center(lafs1)[0]
            matched_kpts0 = (
                centers0[matches[:, 0]].detach().cpu().numpy()
                if matches.numel() > 0
                else np.empty((0, 2))
            )
            matched_kpts1 = (
                centers1[matches[:, 1]].detach().cpu().numpy()
                if matches.numel() > 0
                else np.empty((0, 2))
            )

        elif kind == "sift-lightglue":
            sift = components["sift"]
            lightglue: KF.LightGlueMatcher = components["lightglue"]
            lafs0, _, desc0 = sift(gray0)
            lafs1, _, desc1 = sift(gray1)
            _, matches = lightglue(
                desc0[0],
                desc1[0],
                lafs0,
                lafs1,
                hw1=(sar_rgb.shape[0], sar_rgb.shape[1]),
                hw2=(optical_rgb.shape[0], optical_rgb.shape[1]),
            )
            centers0 = KF.get_laf_center(lafs0)[0]
            centers1 = KF.get_laf_center(lafs1)[0]
            matched_kpts0 = (
                centers0[matches[:, 0]].detach().cpu().numpy()
                if matches.numel() > 0
                else np.empty((0, 2))
            )
            matched_kpts1 = (
                centers1[matches[:, 1]].detach().cpu().numpy()
                if matches.numel() > 0
                else np.empty((0, 2))
            )

        elif kind == "disk-lightglue":
            disk: KF.DISK = components["disk"]
            lightglue = components["lightglue"]
            max_num_keypoints = int(components["max_num_keypoints"])
            feats0 = disk(sar_tensor, n=max_num_keypoints)[0]
            feats1 = disk(optical_tensor, n=max_num_keypoints)[0]
            lafs0 = KF.laf_from_center_scale_ori(
                feats0.keypoints.view(1, -1, 2),
                torch.ones(1, feats0.keypoints.shape[0], 1, 1, device=feats0.keypoints.device),
                torch.zeros(1, feats0.keypoints.shape[0], 1, device=feats0.keypoints.device),
            )
            lafs1 = KF.laf_from_center_scale_ori(
                feats1.keypoints.view(1, -1, 2),
                torch.ones(1, feats1.keypoints.shape[0], 1, 1, device=feats1.keypoints.device),
                torch.zeros(1, feats1.keypoints.shape[0], 1, device=feats1.keypoints.device),
            )
            _, matches = lightglue(
                feats0.descriptors,
                feats1.descriptors,
                lafs0,
                lafs1,
                hw1=(sar_rgb.shape[0], sar_rgb.shape[1]),
                hw2=(optical_rgb.shape[0], optical_rgb.shape[1]),
            )
            matched_kpts0 = (
                feats0.keypoints[matches[:, 0]].detach().cpu().numpy()
                if matches.numel() > 0
                else np.empty((0, 2))
            )
            matched_kpts1 = (
                feats1.keypoints[matches[:, 1]].detach().cpu().numpy()
                if matches.numel() > 0
                else np.empty((0, 2))
            )

        elif kind == "doghardnet-lightglue":
            hardnet: KF.HesAffNetHardNet = components["hardnet"]
            lightglue = components["lightglue"]
            lafs0, _, desc0 = hardnet(gray0)
            lafs1, _, desc1 = hardnet(gray1)
            _, matches = lightglue(
                desc0[0],
                desc1[0],
                lafs0,
                lafs1,
                hw1=(sar_rgb.shape[0], sar_rgb.shape[1]),
                hw2=(optical_rgb.shape[0], optical_rgb.shape[1]),
            )
            centers0 = KF.get_laf_center(lafs0)[0]
            centers1 = KF.get_laf_center(lafs1)[0]
            matched_kpts0 = (
                centers0[matches[:, 0]].detach().cpu().numpy()
                if matches.numel() > 0
                else np.empty((0, 2))
            )
            matched_kpts1 = (
                centers1[matches[:, 1]].detach().cpu().numpy()
                if matches.numel() > 0
                else np.empty((0, 2))
            )

        else:
            msg = f"Unknown Kornia matcher kind: {kind}"
            raise ValueError(msg)

    num_matches = int(matched_kpts0.shape[0])
    model, inlier_mask = fit_geometric_model(
        matched_kpts0,
        matched_kpts1,
        geometry_model="affine",
        ransac_reproj_threshold=3.0,
    )
    if model is None or inlier_mask is None:
        num_inliers = 0
    else:
        num_inliers = int(np.sum(inlier_mask))
    score = float(num_inliers / max(1, num_matches))
    return score, num_inliers, num_matches


def pair_score(
    matcher: Any,
    matcher_backend: str,
    sar_rgb: np.ndarray,
    optical_rgb: np.ndarray,
    normalization: str,
    device: str,
    bf16_on_oom: bool,
) -> tuple[float, int, int]:
    if matcher_backend == "kornia":
        return kornia_pair_score(
            components=matcher,
            sar_rgb=sar_rgb,
            optical_rgb=optical_rgb,
            normalization=normalization,
        )

    sar_tensor = to_matcher_tensor(sar_rgb, normalization, modality="sar")
    optical_tensor = to_matcher_tensor(optical_rgb, normalization, modality="optical")
    result, _ = run_matcher_with_precision(
        matcher=matcher,
        sar_tensor=sar_tensor,
        optical_tensor=optical_tensor,
        device=device,
        bf16_on_oom=bf16_on_oom,
    )

    num_matches = int(np.asarray(result.get("matched_kpts0", np.empty((0, 2)))).shape[0])
    num_inliers = int(result.get("num_inliers", 0))
    score = float(num_inliers / max(1, num_matches))
    return score, num_inliers, num_matches


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = labels[order]
    positives = float(np.sum(y == 1))
    negatives = float(np.sum(y == 0))
    if positives == 0.0 or negatives == 0.0:
        return float("nan")

    tpr = np.cumsum(y == 1) / positives
    fpr = np.cumsum(y == 0) / negatives
    return float(np.trapezoid(tpr, fpr))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = labels[order]
    positives = float(np.sum(y == 1))
    if positives == 0.0:
        return float("nan")

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(1.0, tp + fp)
    recall = tp / positives
    return float(np.trapezoid(precision, recall))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    queries = discover_queries(args.data_dir)
    rng.shuffle(queries)
    queries = queries[: max(1, min(len(queries), int(args.max_queries)))]

    matcher_backend = "vismatch"
    try:
        matcher = vismatch.get_matcher(
            matcher_name=parse_matcher_spec(args.matcher),
            device=args.device,
            max_num_keypoints=args.max_num_keypoints,
        )
    except Exception:
        if not is_kornia_fallback_matcher(args.matcher):
            raise
        matcher_backend = "kornia"
        matcher = build_kornia_components(
            matcher_name=args.matcher,
            device=args.device,
            max_num_keypoints=args.max_num_keypoints,
        )

    all_optical_paths = [path for query in queries for path in query.positive_optical_paths]
    details: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []

    for query in queries:
        sar_rgb = load_sar_mat_as_rgb(query.sar_path)

        positive_set = set(query.positive_optical_paths)
        negative_pool = [path for path in all_optical_paths if path not in positive_set]
        if not negative_pool:
            continue

        neg_count = min(len(negative_pool), int(args.negatives_per_query))
        negative_paths = [
            negative_pool[i] for i in rng.choice(len(negative_pool), size=neg_count, replace=False)
        ]

        candidates = [(path, 1) for path in query.positive_optical_paths] + [
            (path, 0) for path in negative_paths
        ]
        scored_candidates: list[tuple[float, int]] = []

        for optical_path, label in candidates:
            optical_rgb = load_optical_png_rgb(optical_path)
            try:
                score, num_inliers, num_matches = pair_score(
                    matcher=matcher,
                    matcher_backend=matcher_backend,
                    sar_rgb=sar_rgb,
                    optical_rgb=optical_rgb,
                    normalization=args.normalization,
                    device=args.device,
                    bf16_on_oom=args.bf16_on_oom,
                )
                status = "ok"
            except Exception as error:
                score = 0.0
                num_inliers = 0
                num_matches = 0
                status = f"error:{type(error).__name__}"

            scored_candidates.append((score, label))
            details.append(
                {
                    "query_point_id": query.point_id,
                    "sar_path": str(query.sar_path),
                    "optical_path": str(optical_path),
                    "label": label,
                    "score": score,
                    "num_inliers": num_inliers,
                    "num_matches": num_matches,
                    "status": status,
                    "matcher": args.matcher,
                    "normalization": args.normalization,
                }
            )

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        labels_ranked = np.array([label for _, label in scored_candidates], dtype=np.int32)
        retrieval_rows.append(
            {
                "query_point_id": query.point_id,
                "recall_at_1": float(np.any(labels_ranked[:1] == 1)),
                "recall_at_5": float(np.any(labels_ranked[:5] == 1)),
                "recall_at_10": float(np.any(labels_ranked[:10] == 1)),
            }
        )

    details_frame = pd.DataFrame.from_records(details)
    labels = details_frame["label"].to_numpy(dtype=np.int32)
    scores = details_frame["score"].to_numpy(dtype=np.float64)

    retrieval_frame = pd.DataFrame.from_records(retrieval_rows)
    summary = pd.DataFrame.from_records(
        [
            {
                "matcher": args.matcher,
                "normalization": args.normalization,
                "queries": int(retrieval_frame.shape[0]),
                "pairs": int(details_frame.shape[0]),
                "auroc": auroc(labels, scores),
                "auprc": average_precision(labels, scores),
                "recall_at_1": float(retrieval_frame["recall_at_1"].mean())
                if not retrieval_frame.empty
                else float("nan"),
                "recall_at_5": float(retrieval_frame["recall_at_5"].mean())
                if not retrieval_frame.empty
                else float("nan"),
                "recall_at_10": float(retrieval_frame["recall_at_10"].mean())
                if not retrieval_frame.empty
                else float("nan"),
                "mean_score_pos": float(
                    details_frame.loc[details_frame["label"] == 1, "score"].mean()
                ),
                "mean_score_neg": float(
                    details_frame.loc[details_frame["label"] == 0, "score"].mean()
                ),
            }
        ]
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    details_frame.to_csv(args.output_csv, index=False)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)

    print(summary.to_string(index=False))
    print(f"\nWrote SARptical details to {args.output_csv}")
    print(f"Wrote SARptical summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
