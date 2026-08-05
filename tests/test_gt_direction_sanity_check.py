from pathlib import Path

import numpy as np

from rsim.gt_direction_sanity_check import direction_metrics, load_homography, project_points


def test_load_homography_accepts_affine(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("1 0 10\n0 1 20\n", encoding="utf-8")
    homography = load_homography(path)
    assert homography.shape == (3, 3)
    assert np.allclose(homography[2], np.array([0.0, 0.0, 1.0]))


def test_project_points_translation() -> None:
    points = np.array([[0.0, 0.0], [2.0, 3.0]], dtype=np.float64)
    homography = np.array(
        [
            [1.0, 0.0, 5.0],
            [0.0, 1.0, -2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    projected = project_points(points, homography)
    assert np.allclose(projected, np.array([[5.0, -2.0], [7.0, 1.0]], dtype=np.float64))


def test_direction_metrics_prefers_forward() -> None:
    src = np.array([[1.0, 1.0], [3.0, 2.0], [5.0, 4.0]], dtype=np.float64)
    homography = np.array(
        [
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 4.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dst = project_points(src, homography)
    metrics = direction_metrics(src, dst, homography)
    assert metrics["status"] == "ok"
    assert metrics["best_direction"] == "forward"
    assert metrics["forward_median_error"] < metrics["reverse_median_error"]
