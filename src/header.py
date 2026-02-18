import sys
import polars as pl

from netCDF4 import Dataset

from loader import load_dataset


def read_variables(ds: Dataset, verbose: bool = False) -> pl.DataFrame:
    """Build a Polars DataFrame describing every variable in a NetCDF Dataset."""
    rows = []
    for name, var in ds.variables.items():
        row = {
            "variable": name,
            "long_name": getattr(var, "long_name", ""),
            "units": getattr(var, "units", ""),
        }
        if verbose:
            row["dtype"] = str(var.dtype)
            row["shape"] = str(var.shape)
        rows.append(row)
    return pl.DataFrame(rows)


def read_globals(ds: Dataset, verbose: bool = False) -> pl.DataFrame:
    """Build a Polars DataFrame of all global attributes in a NetCDF Dataset."""
    rows = []
    for attr in ds.ncattrs():
        val = getattr(ds, attr)
        row = {
            "attribute": attr,
            "value": str(val),
        }
        if verbose:
            row["type"] = type(val).__name__
        rows.append(row)
    return pl.DataFrame(rows)


def print_variables(ds: Dataset, verbose: bool = False) -> None:
    df = read_variables(ds, verbose)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=60):
        print(df)


def print_globals(ds: Dataset, verbose: bool = False) -> None:
    df = read_globals(ds, verbose)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=120):
        print(df)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python header.py <dataset_id> <filename>")
        print("  e.g: python header.py 638-001 RF01")
        sys.exit(1)

    dataset_id, filename = sys.argv[1], sys.argv[2]
    ds = load_dataset(dataset_id, filename)
    print_globals(ds)
    print_variables(ds)
    ds.close()
