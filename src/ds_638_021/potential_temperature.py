import numpy as np
import polars as pl

from nc.flights import VERTICAL_LEGS
from nc.loader import open_dataset
from nc.units import M_PER_KM
from nc.vars import DS_638_001 as v001

P_850 = 850  # hPa


def compute_theta_850(flight: str) -> dict:
    """
    Compute theta_850 from in-situ vertical legs adjacent to each low-level leg.

    For each low-level leg, extracts the descent leg before and ascent leg after,
    smooths theta over 10 seconds, finds the point closest to 850 hPa, and
    returns the mean theta and altitude from both legs.

    Returns dict keyed by (low_start, low_end) with:
      theta_850     : mean potential temperature at ~850 hPa (K)
      h_850         : mean altitude of ~850 hPa level (km)
      theta_850_std : std between descent and ascent values (K)
      leg_thetas    : individual theta values [descent, ascent]
      leg_times     : datetime64 timestamps of each measurement
    """
    from ds_638_038.segments import load_flight_segments

    segments = load_flight_segments(flight)

    SMOOTH_WINDOW = 10  # seconds (data is 1 Hz, so 10 points)
    P_TOLERANCE = (
        50  # hPa: skip legs that are always more than P_TOLERANCE away from P_850
    )

    with open_dataset(v001.dataset, flight) as ds:
        df = pl.DataFrame(
            {
                "time": ds[v001.time].values,
                "theta": ds[v001.theta].values.astype(np.float64),
                "pressure": ds[v001.pressure].values.astype(np.float64),
                "alt_km": ds[v001.altitude].values.astype(np.float64) / M_PER_KM,
            }
        )

    results = {}

    for low_level_leg, (descent_leg, ascent_leg) in VERTICAL_LEGS[flight].items():
        leg_thetas = []
        leg_alts = []
        leg_times = []

        for leg_start, leg_end in [descent_leg, ascent_leg]:
            seg_times = segments.segment_times(leg_start, leg_end)
            t_start, t_end = seg_times[0], seg_times[-1]

            leg_df = df.filter((pl.col("time") >= t_start) & (pl.col("time") <= t_end))
            if leg_df.height < SMOOTH_WINDOW:
                continue

            leg_df = (
                leg_df.with_columns(
                    pl.col("theta")
                    .rolling_mean(window_size=SMOOTH_WINDOW, center=True)
                    .fill_null(strategy="backward")
                    .fill_null(strategy="forward")
                    .alias("theta_smooth"),
                    (pl.col("pressure") - P_850).abs().alias("p_diff"),
                )
                .filter(pl.col("p_diff") <= P_TOLERANCE)
                .sort("p_diff")
            )

            if leg_df.height == 0:
                continue

            best = leg_df.row(0, named=True)
            leg_thetas.append(best["theta_smooth"])
            leg_alts.append(best["alt_km"])
            leg_times.append(best["time"])

        if leg_thetas:
            results[low_level_leg] = {
                "theta_850": float(np.mean(leg_thetas)),
                "h_850": float(np.mean(leg_alts)),
                "theta_850_std": float(np.std(leg_thetas))
                if len(leg_thetas) > 1
                else np.nan,
                "leg_thetas": leg_thetas,
                "leg_times": leg_times,
            }

    return results
