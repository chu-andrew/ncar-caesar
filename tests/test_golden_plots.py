"""
Golden tests for PNG output reproducibility.

Re-runs each plotting script and compares output PNGs against the golden
reference files in output/ using pixel-level numpy comparison.
"""

import importlib
import os

import numpy as np
import pytest
from PIL import Image

from conftest import GOLDEN_DIR

PLOT_SCRIPTS = [
    ("summary", "638-001/plots/summary"),
    ("segments", "638-038/plots/segments"),
    ("water_path", "638-038/plots/water_path"),
]


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


@pytest.mark.parametrize(
    "module_name, golden_subdir",
    PLOT_SCRIPTS,
    ids=[name for name, _ in PLOT_SCRIPTS],
)
def test_plots_match_golden(module_name, golden_subdir, tmp_output, monkeypatch):
    golden_dir = os.path.join(GOLDEN_DIR, golden_subdir)
    golden_pngs = _collect_pngs(golden_dir)
    if not golden_pngs:
        pytest.skip(f"No golden PNGs found in {golden_dir}")

    tmp_plots = os.path.join(tmp_output, golden_subdir)
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "PLOTS_DIR", tmp_plots)
    module.main()

    actual_pngs = _collect_pngs(tmp_plots)
    assert set(golden_pngs) == set(actual_pngs), (
        f"PNG sets differ.\n  Golden: {golden_pngs}\n  Actual: {actual_pngs}"
    )

    for png in golden_pngs:
        _assert_images_match(
            os.path.join(golden_dir, png),
            os.path.join(tmp_plots, png),
        )
