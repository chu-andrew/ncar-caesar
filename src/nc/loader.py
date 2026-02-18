import os
from glob import glob

from netCDF4 import Dataset

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
    """List available .nc filenames in a dataset, sorted."""
    pattern = os.path.join(get_dataset_dir(dataset_id), "*.nc")
    return sorted(os.path.basename(f) for f in glob(pattern))


def get_file_path(dataset_id: str, filename: str) -> str:
    """Resolve the full path to a .nc file.

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

    raise FileNotFoundError(f"No .nc file matching '{filename}' in {data_dir}")


def load_dataset(dataset_id: str, filename: str) -> Dataset:
    """Open a dataset's NetCDF file and return the Dataset handle."""
    path = get_file_path(dataset_id, filename)
    return Dataset(path)
