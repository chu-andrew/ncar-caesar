"""
Golden tests for CSV output reproducibility.

Re-runs the variables export scripts and compares output byte-for-byte
against the golden reference files in output/.
"""

import os

import pytest

from conftest import GOLDEN_DIR
from nc.variables import export_variable_groups


DATASETS = ["638-001", "638-038"]


def _collect_csv_files(base_dir: str) -> list[str]:
    """Return sorted list of CSV file paths relative to base_dir."""
    csv_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.relpath(os.path.join(root, f), base_dir))
    return sorted(csv_files)


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_variable_csvs_match_golden(dataset_id, tmp_output):
    """Re-export variable CSVs and compare against golden copies."""
    golden_variables_dir = os.path.join(GOLDEN_DIR, dataset_id, "variables")
    if not os.path.isdir(golden_variables_dir):
        pytest.skip(f"No golden variables directory for {dataset_id}")

    output_dir = os.path.join(tmp_output, dataset_id, "variables")
    export_variable_groups(dataset_id, output_dir)

    golden_csvs = _collect_csv_files(golden_variables_dir)
    actual_csvs = _collect_csv_files(output_dir)

    assert golden_csvs == actual_csvs, (
        f"CSV file sets differ.\n  Golden: {golden_csvs}\n  Actual: {actual_csvs}"
    )

    for csv_rel in golden_csvs:
        golden_path = os.path.join(golden_variables_dir, csv_rel)
        actual_path = os.path.join(output_dir, csv_rel)

        golden_bytes = open(golden_path, "rb").read()
        actual_bytes = open(actual_path, "rb").read()

        assert golden_bytes == actual_bytes, (
            f"CSV mismatch for {csv_rel} in dataset {dataset_id}"
        )
