import glob
import os

import numpy as np

from nc.cache import MEMORY
from nc.flights import FLIGHTS
from nc.loader import DATA_DIR
from nc.vars import DS_638_052 as v


@MEMORY.cache
def load_cloud_base(flight: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load cloud base height, in meters, for a flight.
    """
    import xarray as xr

    date_str = FLIGHTS[flight].replace("-", "")
    path = os.path.join(DATA_DIR, v.dataset, "data")
    pattern = os.path.join(path, f"RSmerged.{date_str}_*_L3_CAESAR.nc")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No 638-052 files for {flight} (date {date_str})")

    all_time = []
    all_cb = []

    for f in files:
        ds = xr.open_dataset(f)
        try:
            t = ds[v.time].values  # datetime64[ns]
            cb = ds[v.cloud_base].values  # meters, float32
        finally:
            ds.close()

        # convert datetime64 to fractional hours UTC
        midnight = t.astype("datetime64[D]")
        hours = (t - midnight) / np.timedelta64(1, "h")

        all_time.append(hours)
        all_cb.append(cb.astype(np.float64))

    return np.concatenate(all_time), np.concatenate(all_cb)
