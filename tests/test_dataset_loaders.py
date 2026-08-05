from pathlib import Path

from rsim.dataset_loaders import discover_srif_pairs


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_discover_srif_pairs(tmp_path: Path) -> None:
    root = tmp_path / "srif"
    touch(root / "Optical-SAR" / "pair1_1.jpg")
    touch(root / "Optical-SAR" / "pair1_2.jpg")
    touch(root / "Optical-SAR" / "gt_1.txt")
    touch(root / "Optical-SAR" / "pair2_1.jpg")

    pairs = discover_srif_pairs(root)
    assert len(pairs) == 1
    assert pairs[0].pair_id == "1"
    assert pairs[0].subdataset == "Optical-SAR"
