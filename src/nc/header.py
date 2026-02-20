import os
import sys

import polars as pl
import xarray as xr

from nc.loader import open_dataset


def read_variables(ds: xr.Dataset, verbose: bool = False) -> pl.DataFrame:
    """Build a Polars DataFrame describing every variable in an xarray Dataset."""
    rows = []
    for name in ds.variables:
        var = ds[name]

        units = var.attrs.get("units", var.encoding.get("units", ""))
        long_name = var.attrs.get("long_name", var.encoding.get("long_name", ""))

        row = {
            "variable": name,
            "long_name": long_name,
            "units": units,
        }
        if verbose:
            row["dtype"] = str(var.dtype)
            row["shape"] = str(var.shape)
        rows.append(row)
    return pl.DataFrame(rows)


def read_globals(ds: xr.Dataset, verbose: bool = False) -> pl.DataFrame:
    """Build a Polars DataFrame of all global attributes in an xarray Dataset."""
    rows = []
    for attr, val in ds.attrs.items():
        row = {
            "attribute": attr,
            "value": str(val),
        }
        if verbose:
            row["type"] = type(val).__name__
        rows.append(row)
    return pl.DataFrame(rows)


def export_csv(df: pl.DataFrame, path: str) -> None:
    """Write a Polars DataFrame to a CSV file, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.write_csv(path)
    print(f"Exported: {path}")


def print_variables(ds: xr.Dataset, verbose: bool = False) -> None:
    df = read_variables(ds, verbose)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=60):
        print(df)


def print_globals(ds: xr.Dataset, verbose: bool = False) -> None:
    df = read_globals(ds, verbose)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=120):
        print(df)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python header.py <dataset_id> <filename>")
        print("  e.g: python header.py 638-001 RF01")
        sys.exit(1)

    dataset_id, filename = sys.argv[1], sys.argv[2]
    with open_dataset(dataset_id, filename) as ds:
        print_globals(ds)
        print_variables(ds)
