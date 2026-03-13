"""
Marine Cold-Air Outbreak index computation and composite plots.

Analysis is limited to low-level legs.
SST from RSTB, theta_850 from interpolated MARLi.
"""

import numpy as np
import polars as pl

from nc.flights import FLIGHTS, LOW_LEVEL_LEGS
from nc.loader import open_dataset
from nc.time import utc_hours_to_datetime64
from nc.vars import DS_638_001 as v001

from ds_638_021.potential_temperature import compute_theta_850
from ds_638_038.load import load_gvr_segment

THETA_TOLERANCE = "30s"


def load_rstb(flight: str) -> pl.DataFrame:
    """
    Load radiometric surface temperature (RSTB) from 638-001.

    RSTB is in degC.
    """
    with open_dataset("638-001", flight) as ds:
        times = ds[v001.time].values
        rstb = ds[v001.surface_temp].values

    return pl.DataFrame(
        {
            "time": times,
            "RSTB": rstb.astype(np.float64),
        }
    )


def load_theta850(flight: str) -> pl.DataFrame:
    """
    Load theta_850 from MARLi (638-021) with datetime64 timestamps.
    Uses regridded + interpolated values from compute_theta_850.
    Includes theta_850_std (std of boundary endpoints for interpolated gaps, NaN for measured).
    Only returns rows with valid (non-NaN) theta_850 values.
    """
    result = compute_theta_850(flight)

    # convert regridded float hours to datetime64 using the flight date
    hours = result["time_regrid_utc_hours"]
    times = utc_hours_to_datetime64(hours, FLIGHTS[flight])

    df = pl.DataFrame(
        {
            "time": times,
            "theta_850": result["theta_850"].astype(np.float64),
            "theta_850_std": result["theta_850_std"].astype(np.float64),
        }
    )
    return df.filter(~pl.col("theta_850").is_nan())


def merge_flight_segment(
    flight: str,
    start_pt: int,
    end_pt: int,
    df_rstb: pl.DataFrame,
    df_theta: pl.DataFrame,
) -> pl.DataFrame:
    """
    Merge GVR, RSTB, and theta_850 for a low-level leg.

    theta_850 comes from the interpolated MARLi time series (which fills gaps
    during low-level legs using the mean of adjoining upper-leg boundary values).
    theta_850_std is the std of those boundary endpoints.
    """

    RSTB_TOLERANCE = "10s"

    df_gvr = load_gvr_segment(flight, start_pt, end_pt)
    if df_gvr.is_empty():
        return pl.DataFrame()

    # join RSTB onto GVR timestamps
    df = df_gvr.sort("time").join_asof(
        df_rstb.sort("time"),
        on="time",
        strategy="nearest",
        tolerance=RSTB_TOLERANCE,
    )

    # join theta_850 + theta_850_std onto GVR timestamps
    df = df.join_asof(
        df_theta.sort("time"),
        on="time",
        strategy="nearest",
        tolerance=THETA_TOLERANCE,
    )

    # compute MCAO index: SST(K) - theta_850; RSTB is in degC
    df = df.with_columns(
        (pl.col("RSTB") + 273.15).alias("SST_K"),
    ).with_columns(
        (pl.col("SST_K") - pl.col("theta_850")).alias("MCAO"),
    )

    df = df.with_columns(
        pl.lit(flight).alias("flight"),
        pl.lit(f"{start_pt}-{end_pt}").alias("segment"),
    )

    return df


def build_merged_dataset() -> pl.DataFrame:
    """Build the full merged dataset across all flights and segments."""
    frames = []

    for flight in LOW_LEVEL_LEGS:
        if flight not in FLIGHTS:
            continue
        print(f"Processing {flight}...")

        df_rstb = load_rstb(flight)
        df_theta = load_theta850(flight)

        # expand multi-segment low-level legs into unit segments
        low_level_units = set()
        for s, e in LOW_LEVEL_LEGS[flight]:
            for j in range(s, e):
                low_level_units.add((j, j + 1))

        for start_pt, end_pt in sorted(low_level_units):
            df_seg = merge_flight_segment(flight, start_pt, end_pt, df_rstb, df_theta)
            if not df_seg.is_empty():
                n_valid = df_seg.filter(
                    pl.col("MCAO").is_not_null() & ~pl.col("MCAO").is_nan()
                ).height
                mean_t = df_seg["theta_850"].mean()
                std_vals = df_seg["theta_850_std"].drop_nulls().drop_nans()
                std_t = std_vals.mean() if len(std_vals) > 0 else float("nan")
                print(
                    f"\tlow-level {start_pt}-{end_pt}: "
                    f"{n_valid}/{df_seg.height} valid MCAO, "
                    f"theta_850={mean_t:.2f} +/- {std_t:.2f} K"
                )
                frames.append(df_seg)

    if not frames:
        raise RuntimeError("no data merged: check segment definitions and data files")

    common_cols = [
        "time",
        "LWP",
        "WVP",
        "SST_K",
        "theta_850",
        "theta_850_std",
        "MCAO",
        "flight",
        "segment",
    ]
    frames = [f.select(common_cols) for f in frames]
    return pl.concat(frames)
