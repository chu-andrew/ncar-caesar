import os
from typing import Tuple

import numpy as np
import polars as pl
import xarray as xr

from ds_638_038.load import load_gvr_segment
from ds_638_038.segments import load_flight_segments
from nc.cache import MEMORY
from nc.flights import FLIGHTS, LOW_LEVEL_LEGS
from nc.loader import PROJECT_ROOT
from nc.time import seconds_to_datetime64
from nc.vars import MICROPHYSICS as vm

# cloud-phase flags
PHASE_CLEAR = 0
PHASE_ICE = 1
PHASE_MIXED = 2
PHASE_LIQUID = 3
PHASE_DRIZZLE = 4

MICROPHYSICS_DATASET_DIR = os.path.join(
    PROJECT_ROOT, "data", "microphysics_beta", "data"
)


def _open_micro(flight: str) -> xr.Dataset:
    path = os.path.join(MICROPHYSICS_DATASET_DIR, f"{flight}_microphysics_beta.nc")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Microphysics file not found: {path}")
    return xr.open_dataset(path)


@MEMORY.cache
def load_microphysics_segment(
    flight: str, start_pt: int, end_pt: int, phase_filter: frozenset[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if flight not in FLIGHTS:
        raise ValueError(f"Unknown flight: {flight}")

    # get segment time bounds
    fs = load_flight_segments(flight)
    seg_slice = fs.segment_slice(start_pt, end_pt)
    t_start = fs.times[seg_slice.start]
    t_stop = fs.times[seg_slice.stop - 1] if seg_slice.stop else fs.times[-1]

    # convert to seconds-from-midnight for slicing microphysics
    flight_date = FLIGHTS[flight]
    base_ns = np.datetime64(flight_date, "ns")

    def _to_sec(t: np.datetime64) -> float:
        return float((t - base_ns) / np.timedelta64(1, "s"))

    sec_start = _to_sec(t_start)
    sec_stop = _to_sec(t_stop)

    # load microphysics
    ds = _open_micro(flight)
    try:
        micro_time = ds[vm.time].values  # seconds from midnight
        cloud_phase = ds[vm.cloud_phase].values  # (time,)
        conc = ds[vm.concentration].values  # (bin_centers, time), #/m^4
        bin_edges_um = ds[vm.bin_edges].values  # um, (170,)
    finally:
        ds.close()

    # time & phase mask
    in_segment = (micro_time >= sec_start) & (micro_time <= sec_stop)
    is_phase = np.isin(cloud_phase, list(phase_filter))
    mask = in_segment & is_phase

    if not mask.any():
        # return empty arrays with correct shapes
        return (
            np.array([], dtype="datetime64[ns]"),
            np.empty((len(bin_edges_um) - 1, 0), dtype=np.float64),
            np.empty(len(bin_edges_um) - 1, dtype=np.float64),
            np.empty(len(bin_edges_um) - 1, dtype=np.float64),
        )

    t_sel = micro_time[mask]
    c_sel = conc[:, mask]  # (n_bins, n_times)

    # compute bin centers and widths
    bin_centers_um = (bin_edges_um[:-1] + bin_edges_um[1:]) / 2
    bin_widths_um = np.diff(bin_edges_um)

    times_dt = seconds_to_datetime64(t_sel, flight_date)

    return times_dt, c_sel, bin_centers_um, bin_widths_um


@MEMORY.cache
def build_low_level_dataset(
    phase_filter: frozenset[int] = frozenset({PHASE_ICE}),
) -> pl.DataFrame:
    """
    Build complete dataset for all low-level legs.
    Loads microphysics size distributions and joins with WVP/LWP from GVR.
    """
    JOIN_TOLERANCE = "2s"

    all_frames: list[pl.DataFrame] = []

    for flight in LOW_LEVEL_LEGS:
        if flight not in FLIGHTS:
            continue

        print(f"Building {flight} dataset")
        for start_pt, end_pt in LOW_LEVEL_LEGS[flight]:
            try:
                # load microphysics
                times, concentration, bin_centers, bin_widths = (
                    load_microphysics_segment(flight, start_pt, end_pt, phase_filter)
                )

                if len(times) == 0:
                    print(
                        f"\tsegment {start_pt}-{end_pt}: no data matching phase filter"
                    )
                    continue

                # load water path
                df_wp = load_gvr_segment(flight, start_pt, end_pt)

                # create df with concentration as list column
                # each row is one timestep
                n_times = len(times)

                df_micro = pl.DataFrame(
                    {
                        "time": times,
                        "concentration": [
                            concentration[:, i].tolist() for i in range(n_times)
                        ],
                        "bin_centers": [bin_centers.tolist()] * n_times,
                        "bin_widths": [bin_widths.tolist()] * n_times,
                        "flight": [flight] * n_times,
                        "segment_id": [f"{start_pt}-{end_pt}"] * n_times,
                    }
                )

                # join water path using join_asof
                df_merged = df_micro.sort("time").join_asof(
                    df_wp.sort("time"),
                    on="time",
                    strategy="nearest",
                    tolerance=JOIN_TOLERANCE,
                )

                # filter out rows with null joins
                df_merged = df_merged.filter(
                    pl.col("LWP").is_not_null() & pl.col("WVP").is_not_null()
                )

                if not df_merged.is_empty():
                    print(
                        f"\tsegment {start_pt}-{end_pt}: {df_merged.height} timesteps"
                    )
                    all_frames.append(df_merged)

            except FileNotFoundError as e:
                print(f"\t{e}")
                break

    if not all_frames:
        raise RuntimeError("No data loaded from any segment.")

    df = pl.concat(all_frames)

    print(f"\nTotal dataset: {df.height} timesteps across {len(all_frames)} segments")

    return df
