"""Group NetCDF files by their variable sets and export a CSV per group."""

import os
import sys

from nc.header import export_csv, read_variables
from nc.loader import PROJECT_ROOT, list_files, load_dataset


def group_by_variables(dataset_id: str) -> dict[frozenset[str], list[str]]:
    """Group .nc files by their variable names."""
    files = list_files(dataset_id)
    groups: dict[frozenset[str], list[str]] = {}
    for f in files:
        ds = load_dataset(dataset_id, f)
        variables = frozenset(ds.variables.keys())
        ds.close()
        groups.setdefault(variables, []).append(f)
    return groups


def export_variable_groups(dataset_id: str, output_dir: str) -> None:
    """
    Export a variables CSV for each unique variable set in a dataset.

    NB: files are grouped by variable names only. Metadata (long_name, units)
    may differ between files in the same group — the CSV uses metadata from the
    first file in each group.
    """
    groups = group_by_variables(dataset_id)
    all_files = list_files(dataset_id)
    print(
        f"Found {len(all_files)} files in {dataset_id}, {len(groups)} unique variable set(s)\n"
    )

    for var_set, filenames in groups.items():
        labels = [
            f.split(".")[0] for f in filenames
        ]  # NB: brittle depending on filename
        label = ",".join(labels)

        ds = load_dataset(dataset_id, filenames[0])
        df = read_variables(ds)
        ds.close()

        csv_path = os.path.join(output_dir, f"variables_{label}.csv")
        export_csv(df, csv_path)
        print(f"\tFiles: {labels} ({len(var_set)} variables)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python variables.py <dataset_id> [output_dir]")
        print("  e.g: python variables.py 638-001")
        print("       python variables.py 638-001 output/my_dir")
        sys.exit(1)

    dataset_id = sys.argv[1]
    output_dir = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.path.join(PROJECT_ROOT, "output", dataset_id, "variables")
    )
    export_variable_groups(dataset_id, output_dir)
