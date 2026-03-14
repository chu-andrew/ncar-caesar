import numpy as np

import metpy.calc as mpcalc
from metpy.units import units as munits

from nc.flights import MARLI_FILES
from nc.loader import open_dataset
from nc.vars import DS_638_021 as v

P_850 = 850  # hPa

MAD_K = 5.0  # multiplier for MAD
TEMPORAL_RESOLUTION = 30  # seconds


def mask_temperature_outliers(T: np.ndarray, k: float = MAD_K) -> np.ndarray:
    """
    Mask outlier temperatures using per-level median absolute deviation.
    """

    T = T.astype(np.float64, copy=True)
    T[(T >= 9999.0) | (T < -100) | (T > 100)] = np.nan

    if T.ndim == 1:
        T = T[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    for j in range(T.shape[1]):
        col = T[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) < 3:
            continue
        median = np.median(valid)
        mad = np.median(np.abs(valid - median))
        if mad == 0:
            continue
        else:
            T[np.abs(col - median) > k * mad, j] = np.nan

    return T[:, 0] if squeeze else T


def height_to_pressure(h_km: np.ndarray) -> np.ndarray:
    """Convert height (km MSL) to pressure (hPa) using standard atmosphere."""
    return mpcalc.height_to_pressure_std(h_km * munits.km).to("hPa").magnitude


def potential_temperature(t_celsius: np.ndarray, p_hpa: float) -> np.ndarray:
    """Potential temperature from temperature (degC) and pressure (hPa)."""
    T = t_celsius * munits.degC
    P = p_hpa * munits.hPa
    return mpcalc.potential_temperature(P, T).to("kelvin").magnitude


def _extract_theta_850(ds) -> tuple[np.ndarray, np.ndarray, float, float]:
    H = ds["H"].values  # bin heights (MSL), km
    T = ds["T"].values  # atmospheric temperature, degC (n_times x n_bins)
    time = ds["time"].values  # UTC time

    # convert all height bins to pressure
    p_levels = height_to_pressure(H)

    # find bin closest to 850 hPa
    idx = np.argmin(np.abs(p_levels - P_850))
    h_actual = float(H[idx])
    p_actual = float(p_levels[idx])

    # extract temperature time-series at that level, mask outliers
    t_at_level = mask_temperature_outliers(T[:, idx])

    theta = potential_temperature(t_at_level, p_actual)
    return time, theta, h_actual, p_actual


def regrid_timeseries(
    time_hours: np.ndarray,
    data: np.ndarray,
    dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Regrid time series to uniform temporal resolution (of dt_seconds) using block averaging.
    """
    dt_hours = dt_seconds / 3600.0

    # create uniform grid from min to max time
    t_min = np.nanmin(time_hours)
    t_max = np.nanmax(time_hours)
    time_edges = np.arange(t_min, t_max + dt_hours, dt_hours)
    time_regrid = time_edges[:-1] + dt_hours / 2  # bin centers

    # bin data and compute mean in each bin
    data_regrid = np.full(len(time_regrid), np.nan)
    for i in range(len(time_regrid)):
        mask = (time_hours >= time_edges[i]) & (time_hours < time_edges[i + 1])
        bin_data = data[mask]
        valid_data = bin_data[~np.isnan(bin_data)]

        if len(valid_data) > 0:
            data_regrid[i] = np.mean(valid_data)

    return time_regrid, data_regrid


def find_gaps(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Find contiguous regions of True values in a boolean mask.

    Returns list of (start, end) indices for each gap.
    """
    gaps = []
    in_gap = False
    gap_start = 0

    for i, is_missing in enumerate(mask):
        if is_missing and not in_gap:
            gap_start = i
            in_gap = True
        elif not is_missing and in_gap:
            gaps.append((gap_start, i))
            in_gap = False

    if in_gap:
        gaps.append((gap_start, len(mask)))

    return gaps


def interpolate_gaps(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fill interior gaps using the mean of the two boundary endpoints.

    Returns:
        data_filled: array with interior gaps filled (constant per gap)
        is_interpolated: boolean mask indicating which points were filled
        interp_std: std of the two boundary endpoints (NaN where not interpolated)
    """
    missing = np.isnan(data)
    data_filled = data.copy()
    interp_std = np.full_like(data, np.nan)
    valid = ~missing

    if np.sum(valid) < 2:
        return data, missing, interp_std

    gaps = find_gaps(missing)
    is_interpolated = np.zeros_like(data, dtype=bool)

    for gap_start, gap_end in gaps:
        # only fill interior gaps (valid data on both sides)
        if not np.any(valid[:gap_start]) or not np.any(valid[gap_end:]):
            continue

        # endpoints: last valid before gap and first valid after gap
        y_before = data[np.where(valid[:gap_start])[0][-1]]
        y_after = data[gap_end + np.argmax(valid[gap_end:])]

        gap_mean = (y_before + y_after) / 2.0
        gap_std = np.std([y_before, y_after])

        data_filled[gap_start:gap_end] = gap_mean
        is_interpolated[gap_start:gap_end] = True
        interp_std[gap_start:gap_end] = gap_std

    return data_filled, is_interpolated, interp_std


def compute_theta_850(flight: str, interpolate: bool = True) -> dict:
    filenames = MARLI_FILES[flight]

    all_time = []
    all_theta = []
    all_alt = []
    h_actual = None
    p_actual = None

    for filename in filenames:
        with open_dataset(v.dataset, filename) as ds:
            time, theta, h, p = _extract_theta_850(ds)
            all_alt.append(ds[v.altitude].values)
        all_time.append(time)
        all_theta.append(theta)
        if h_actual is None:
            h_actual = h
            p_actual = p

    time_hours_native = np.concatenate(all_time)
    theta_850_native = np.concatenate(all_theta)
    altitude_native = np.concatenate(all_alt)

    time_hours_regrid, theta_850_regrid = regrid_timeseries(
        time_hours_native, theta_850_native, TEMPORAL_RESOLUTION
    )

    # interpolate gaps if requested
    if interpolate:
        theta_850_filled, is_interpolated, interp_std = interpolate_gaps(
            theta_850_regrid
        )
    else:
        theta_850_filled = theta_850_regrid
        is_interpolated = np.zeros_like(theta_850_regrid, dtype=bool)
        interp_std = np.full_like(theta_850_regrid, np.nan)

    return {
        "time_utc_hours": time_hours_native,
        "theta_850": theta_850_filled,
        "time_regrid_utc_hours": time_hours_regrid,
        "altitude": altitude_native,
        "h_850": h_actual,
        "p_850": p_actual,
        "is_interpolated": is_interpolated,
        "theta_850_std": interp_std,
    }
