"""Group NetCDF files by their variable sets and export a CSV per group."""

import hashlib
import os
import sys
from pathlib import Path

import polars as pl

from nc.header import export_csv, read_variables
from nc.loader import PROJECT_ROOT, get_dataset_dir, list_dir_files, open_file


def group_by_variables(files: list[Path]) -> dict[frozenset[str], list[Path]]:
    """Group NetCDF files by their variable names."""
    groups: dict[frozenset[str], list[Path]] = {}
    for f in files:
        with open_file(f, decode_times=False) as ds:
            variables = frozenset(set(ds.data_vars) | set(ds.coords))
        groups.setdefault(variables, []).append(f)
    return groups


def _make_csv_name(labels: list[str]) -> str:
    label = ",".join(labels)
    csv_name = f"variables_{label}.csv"
    if len(csv_name.encode()) > 255:
        digest = hashlib.sha256(label.encode()).hexdigest()[:5]
        csv_name = f"variables_{labels[0]}_+{len(labels) - 1}_{digest}.csv"
    return csv_name


def export_variable_groups(files: list[Path], output_dir: str) -> None:
    """
    Export a variables CSV for each unique variable set among the given files.

    NB: files are grouped by variable names only. Metadata (long_name, units)
    may differ between files in the same group — the CSV uses metadata from the
    first file in each group.
    """
    groups: dict[frozenset[str], list[Path]] = {}
    first_vars: dict[frozenset[str], pl.DataFrame] = {}

    for f in files:
        with open_file(f, decode_times=False) as ds:
            variables = frozenset(set(ds.data_vars) | set(ds.coords))
            if variables not in first_vars:
                first_vars[variables] = read_variables(ds)
        groups.setdefault(variables, []).append(f)

    print(f"Found {len(files)} files, {len(groups)} unique variable set(s)\n")

    for var_set, paths in groups.items():
        labels = [p.stem for p in paths]
        csv_path = os.path.join(output_dir, _make_csv_name(labels))
        export_csv(first_vars[var_set], csv_path)
        print(f"\tFiles: {labels} ({len(var_set)} variables)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python variables.py <dataset_id_or_path> [output_dir]")
        print("  e.g: python variables.py 638-001")
        print("       python variables.py 638-001 output/my_dir")
        sys.exit(1)

    arg = sys.argv[1]

    if os.path.isabs(arg):
        directory = Path(arg)
        output_dir = (
            sys.argv[2]
            if len(sys.argv) > 2
            else os.path.join(
                PROJECT_ROOT,
                "output",
                "remote",
                str(directory.name).lower(),
                "variables",
            )
        )
    else:
        directory = get_dataset_dir(arg)
        output_dir = (
            sys.argv[2]
            if len(sys.argv) > 2
            else os.path.join(PROJECT_ROOT, "output", arg, "variables")
        )

    export_variable_groups(list_dir_files(directory), output_dir)
