"""
Golden tests for PNG output reproducibility.

Re-runs each plotting script and compares output PNGs against the golden
reference files in output/ using pixel-level numpy comparison.
"""

import os
import sys

import numpy as np
import pytest
from PIL import Image

from conftest import GOLDEN_DIR, PROJECT_ROOT

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "638-001"))
sys.path.insert(0, os.path.join(SRC_DIR, "638-038"))


def _png_to_array(path: str) -> np.ndarray:
    """Load a PNG file as a numpy uint8 array."""
    return np.array(Image.open(path))


def _assert_images_match(golden_path: str, actual_path: str, tolerance: float = 0.0):
    """Compare two PNG images. Allows a small fraction of differing pixels."""
    golden = _png_to_array(golden_path)
    actual = _png_to_array(actual_path)

    assert golden.shape == actual.shape, (
        f"Image shape mismatch: golden {golden.shape} vs actual {actual.shape}"
    )

    if tolerance == 0.0:
        assert np.array_equal(golden, actual), (
            f"Images differ: {os.path.basename(golden_path)}"
        )
    else:
        total_pixels = golden.size
        diff_pixels = np.count_nonzero(golden != actual)
        frac = diff_pixels / total_pixels
        assert frac <= tolerance, (
            f"Images differ by {frac:.4%} (tolerance {tolerance:.4%}): "
            f"{os.path.basename(golden_path)}"
        )


def _collect_pngs(directory: str) -> list[str]:
    """Return sorted list of PNG filenames in a directory."""
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.endswith(".png"))


def _assert_golden_pngs(golden_dir: str, actual_dir: str):
    """Assert that actual PNGs match golden PNGs exactly."""
    golden_pngs = _collect_pngs(golden_dir)
    if not golden_pngs:
        pytest.skip(f"No golden PNGs found in {golden_dir}")

    actual_pngs = _collect_pngs(actual_dir)
    assert set(golden_pngs) == set(actual_pngs), (
        f"PNG sets differ.\n  Golden: {golden_pngs}\n  Actual: {actual_pngs}"
    )

    for png in golden_pngs:
        _assert_images_match(
            os.path.join(golden_dir, png),
            os.path.join(actual_dir, png),
        )


class TestSummaryPlots:
    """Golden tests for 638-001 summary plots."""

    GOLDEN_PLOTS_DIR = os.path.join(GOLDEN_DIR, "638-001/plots/summary")

    def test_summary_plots_match_golden(self, tmp_output, monkeypatch):
        tmp_plots = os.path.join(tmp_output, "638-001/plots/summary")

        import summary

        monkeypatch.setattr(summary, "PLOTS_DIR", tmp_plots)
        summary.main()

        _assert_golden_pngs(self.GOLDEN_PLOTS_DIR, tmp_plots)


class TestSegmentPlots:
    """Golden tests for 638-038 segment plots."""

    GOLDEN_PLOTS_DIR = os.path.join(GOLDEN_DIR, "638-038/plots/segments")

    def test_segment_plots_match_golden(self, tmp_output, monkeypatch):
        tmp_plots = os.path.join(tmp_output, "638-038/plots/segments")

        import segments

        monkeypatch.setattr(segments, "PLOTS_DIR", tmp_plots)
        segments.main()

        _assert_golden_pngs(self.GOLDEN_PLOTS_DIR, tmp_plots)


class TestWaterPathPlots:
    """Golden tests for 638-038 water path plots."""

    GOLDEN_PLOTS_DIR = os.path.join(GOLDEN_DIR, "638-038/plots/water_path")

    def test_water_path_plots_match_golden(self, tmp_output, monkeypatch):
        tmp_plots = os.path.join(tmp_output, "638-038/plots/water_path")

        import water_path

        monkeypatch.setattr(water_path, "PLOTS_DIR", tmp_plots)
        water_path.main()

        _assert_golden_pngs(self.GOLDEN_PLOTS_DIR, tmp_plots)
