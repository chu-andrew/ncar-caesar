import numpy as np
import polars as pl

from nc.cache import MEMORY
from nc.loader import open_dataset
from nc.vars import DS_638_038 as v
from ds_638_038.segments import load_flight_segments


@MEMORY.cache
def load_gvr_segment(flight: str, start_pt: int, end_pt: int) -> pl.DataFrame:
    fs = load_flight_segments(flight)
    s = fs.segment_slice(start_pt, end_pt)

    with open_dataset(v.dataset, flight) as ds:
        times = ds[v.time].values[s]
        lwp = ds[v.lwp].values[s]
        wvp = ds[v.wvp].values[s]
        alt = ds[v.altitude].values[s]

    return pl.DataFrame(
        {
            "time": times,
            "LWP": lwp.astype(np.float64),
            "WVP": wvp.astype(np.float64),
            "alt": alt.astype(np.float64),
        }
    )
