import numpy as np
import polars as pl

from nc.loader import open_dataset
from ds_638_038.segments import load_flight_segments

DATASET = "638-038"
TIME = "time"
ALTITUDE = "alt"


def load_gvr_segment(flight: str, start_pt: int, end_pt: int) -> pl.DataFrame:
    fs = load_flight_segments(flight)
    s = fs.segment_slice(start_pt, end_pt)

    with open_dataset(DATASET, flight) as ds:
        times = ds[TIME].values[s]
        lwp = ds["LWP"].values[s]
        wvp = ds["WVP"].values[s]
        alt = ds[ALTITUDE].values[s]

    return pl.DataFrame(
        {
            "time": times,
            "LWP": lwp.astype(np.float64),
            "WVP": wvp.astype(np.float64),
            "alt": alt.astype(np.float64),
        }
    )
