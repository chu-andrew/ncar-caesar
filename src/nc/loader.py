import os
from contextlib import contextmanager
from pathlib import Path

import xarray as xr

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

NC_EXTENSIONS = ("*.nc", "*.cdf")


def list_dir_files(directory: Path | str) -> list[Path]:
    """List NetCDF files (.nc, .cdf) in a directory, sorted."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = []
    for ext in NC_EXTENSIONS:
        files.extend(directory.glob(ext))
    return sorted(files)


@contextmanager
def open_file(path: Path | str, **kwargs):
    """Context manager that opens a NetCDF file by path."""
    kwargs.setdefault("decode_cf", True)
    kwargs.setdefault("decode_timedelta", True)
    kwargs.setdefault("engine", "netcdf4")
    with xr.open_dataset(Path(path), **kwargs) as ds:
        yield ds


def get_dataset_dir(dataset_id: str) -> Path:
    """Return the data subdirectory for a given dataset."""
    path = Path(DATA_DIR) / dataset_id / "data"
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {path}")
    return path


def list_files(dataset_id: str) -> list[str]:
    """List available NetCDF filenames (.nc, .cdf) in a dataset, sorted."""
    return [p.name for p in list_dir_files(get_dataset_dir(dataset_id))]


def get_file_path(dataset_id: str, filename: str) -> Path:
    """Resolve the full path to a NetCDF file (.nc or .cdf).

    ``filename`` can be a prefix like 'RF01' or a full filename.
    """
    data_dir = get_dataset_dir(dataset_id)

    exact = data_dir / filename
    if exact.is_file():
        return exact

    matches = list(data_dir.glob(f"{filename}*"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous prefix '{filename}', matches: {[m.name for m in matches]}"
        )

    raise FileNotFoundError(f"No NetCDF file matching '{filename}' in {data_dir}")


@contextmanager
def open_dataset(dataset_id: str, filename: str):
    """Context manager that opens a NetCDF dataset by dataset_id and filename."""
    with open_file(get_file_path(dataset_id, filename)) as ds:
        yield ds
