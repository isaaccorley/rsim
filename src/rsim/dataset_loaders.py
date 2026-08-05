from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True)
class SrifPairSample:
    case_id: str
    pair_id: str
    subdataset: str
    image1_path: Path
    image2_path: Path
    transform_path: Path


def _find_matching_file(directory: Path, pattern: re.Pattern[str]) -> Path | None:
    for path in sorted(directory.iterdir()):
        if pattern.match(path.name):
            return path
    return None


def discover_srif_pairs(
    data_dir: Path, subdatasets: Sequence[str] | None = None
) -> list[SrifPairSample]:
    image1_pattern = re.compile(r"^pair(?P<pair_id>\d+)_1\.(jpg|jpeg|png)$", flags=re.IGNORECASE)
    image2_template = r"^pair{pair_id}_2\.(jpg|jpeg|png)$"
    transform_template = r"^gt_{pair_id}\.txt$"

    if subdatasets is None:
        selected_subdatasets = [path.name for path in sorted(data_dir.iterdir()) if path.is_dir()]
    else:
        selected_subdatasets = list(subdatasets)

    pairs: list[SrifPairSample] = []
    for subdataset in selected_subdatasets:
        subset_dir = data_dir / subdataset
        if not subset_dir.exists() or not subset_dir.is_dir():
            continue

        for image1_path in sorted(subset_dir.iterdir()):
            match = image1_pattern.match(image1_path.name)
            if match is None:
                continue

            pair_id = match.group("pair_id")
            image2_path = _find_matching_file(
                subset_dir,
                re.compile(image2_template.format(pair_id=pair_id), flags=re.IGNORECASE),
            )
            transform_path = _find_matching_file(
                subset_dir,
                re.compile(transform_template.format(pair_id=pair_id), flags=re.IGNORECASE),
            )
            if image2_path is None or transform_path is None:
                continue

            pairs.append(
                SrifPairSample(
                    case_id=subdataset,
                    pair_id=pair_id,
                    subdataset=subdataset,
                    image1_path=image1_path,
                    image2_path=image2_path,
                    transform_path=transform_path,
                )
            )

    if not pairs:
        msg = f"No valid SRIF pairs found under {data_dir}."
        raise FileNotFoundError(msg)
    return pairs


__all__ = [
    "SrifPairSample",
    "discover_srif_pairs",
]
