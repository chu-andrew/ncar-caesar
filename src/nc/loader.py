import os
from contextlib import contextmanager
from glob import glob

import xarray as xr

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def get_dataset_dir(dataset_id: str) -> str:
    """Return the data subdirectory for a given dataset."""
    path = os.path.join(DATA_DIR, dataset_id, "data")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Dataset directory not found: {path}")
    return path


def list_files(dataset_id: str) -> list[str]:
    """List available NetCDF filenames (.nc, .cdf) in a dataset, sorted."""
    NC_EXTENSIONS = ("*.nc", "*.cdf")

    data_dir = get_dataset_dir(dataset_id)
    files = []
    for ext in NC_EXTENSIONS:
        files.extend(glob(os.path.join(data_dir, ext)))
    return sorted(os.path.basename(f) for f in files)


def get_file_path(dataset_id: str, filename: str) -> str:
    """Resolve the full path to a NetCDF file (.nc or .cdf).

    ``filename`` can be a prefix like 'RF01' or a full filename.
    """
    data_dir = get_dataset_dir(dataset_id)

    # Try exact match first
    exact = os.path.join(data_dir, filename)
    if os.path.isfile(exact):
        return exact

    # Try prefix match
    matches = glob(os.path.join(data_dir, f"{filename}*"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = [os.path.basename(m) for m in matches]
        raise ValueError(f"Ambiguous prefix '{filename}', matches: {names}")

    raise FileNotFoundError(f"No NetCDF file matching '{filename}' in {data_dir}")


@contextmanager
def open_dataset(dataset_id: str, filename: str):
    """Context manager that opens a NetCDF dataset and ensures it is closed."""
    path = get_file_path(dataset_id, filename)
    with xr.open_dataset(path, decode_cf=True, decode_timedelta=True) as ds:
        yield ds
