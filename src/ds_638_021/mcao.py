"""
Marine Cold-Air Outbreak index computation and composite plots.

Analysis is limited to low-level legs.
SST from RSTB, theta_850 from in-situ vertical legs.
"""

import numpy as np
import polars as pl

from nc.cache import MEMORY
from nc.flights import FLIGHTS, LOW_LEVEL_LEGS
from nc.loader import open_dataset
from nc.units import ZERO_CELSIUS_IN_KELVIN
from nc.vars import DS_638_001 as v001

from ds_638_021.potential_temperature import compute_theta_850
from ds_638_038.load import load_gvr_segment


@MEMORY.cache
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
    ).sort("time")


def merge_flight_segment(
    flight: str,
    start_pt: int,
    end_pt: int,
    df_rstb: pl.DataFrame,
    theta_850: float,
    theta_850_std: float,
) -> pl.DataFrame:
    """
    Merge GVR and RSTB for a low-level leg, adding theta_850 from in-situ vertical legs.
    """
    df_gvr = load_gvr_segment(flight, start_pt, end_pt)
    if df_gvr.is_empty():
        return pl.DataFrame()

    df = df_gvr.sort("time").join_asof(
        df_rstb,
        on="time",
        strategy="nearest",
        tolerance="10s",
    )

    df = df.with_columns(
        pl.lit(theta_850).alias("theta_850"),
        pl.lit(theta_850_std).alias("theta_850_std"),
    )

    df = df.with_columns(
        (pl.col("RSTB") + ZERO_CELSIUS_IN_KELVIN).alias("SST_K"),
    ).with_columns(
        (pl.col("SST_K") - pl.col("theta_850")).alias("MCAO"),
    )

    df = df.with_columns(
        pl.lit(flight).alias("flight"),
        pl.lit(f"{start_pt}-{end_pt}").alias("segment"),
    )

    return df


@MEMORY.cache
def build_merged_dataset() -> pl.DataFrame:
    """Build the full merged dataset across all flights and segments."""
    frames = []

    for flight in LOW_LEVEL_LEGS:
        if flight not in FLIGHTS:
            continue
        print(f"Processing {flight}...")

        df_rstb = load_rstb(flight)
        theta_legs = compute_theta_850(flight)

        for low_level_leg in LOW_LEVEL_LEGS[flight]:
            leg = theta_legs.get(low_level_leg, {})
            theta_850 = leg.get("theta_850", float("nan"))
            theta_850_std = leg.get("theta_850_std", float("nan"))

            low_start, low_end = low_level_leg
            for j in range(low_start, low_end):
                df_seg = merge_flight_segment(
                    flight, j, j + 1, df_rstb, theta_850, theta_850_std
                )
                if not df_seg.is_empty():
                    n_valid = df_seg.filter(
                        pl.col("MCAO").is_not_null() & ~pl.col("MCAO").is_nan()
                    ).height
                    print(
                        f"\tlow-level {j}-{j + 1}: "
                        f"{n_valid}/{df_seg.height} valid MCAO, "
                        f"theta_850={theta_850:.2f} +/- {theta_850_std:.2f} K"
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
