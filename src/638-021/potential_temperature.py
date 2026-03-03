"""
Calculate potential temperature at 850 hPa.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from nc.flights import MARLI_FILES
from nc.loader import DATASET_VARS, PROJECT_ROOT, open_dataset

_vars = DATASET_VARS["638-021"]
ALTITUDE = _vars["altitude"]

DATASET = "638-021"
PLOTS_DIR = os.path.join(PROJECT_ROOT, f"output/{DATASET}/plots/potential_temperature")

FILL_VALUE = 9999.0
P_850 = 850  # hPa

MAD_K = 5.0  # multiplier for MAD
TEMPORAL_RESOLUTION = 30 # seconds


def mask_temperature_outliers(T: np.ndarray, k: float = MAD_K) -> np.ndarray:
    """
    Mask outlier temperatures using per-level median absolute deviation.
    """

    T = T.astype(np.float64, copy=True)
    T[(T >= FILL_VALUE) | (T < -100) | (T > 100)] = np.nan

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
    """Convert height (km MSL) to pressure (hPa) using the hypsometric equation.

    p = p_surface * exp(-h * g / (R_d * T_mean))
    """
    P_SURFACE = 1013.25  # hPa, standard sea-level pressure
    T_MEAN = 288.15  # K, standard mean temperature (ISA sea-level) = 15 degC
    R_D = 287.05  # J/(kg K), specific gas constant for dry air
    G = 9.81  # m/s^2, gravitational acceleration

    h_m = h_km * 1000.0  # km -> m
    return P_SURFACE * np.exp(-h_m * G / (R_D * T_MEAN))


def potential_temperature(t_celsius: np.ndarray, p_hpa: float) -> np.ndarray:
    KAPPA = 0.286  # R/c_p

    t_kelvin = t_celsius + 273.15
    return t_kelvin * (1000.0 / p_hpa) ** KAPPA


def _extract_theta_850(ds) -> tuple[np.ndarray, np.ndarray, float, float]:
    H = ds["H"].values  # bin height (MSL), km
    T = ds["T"].values  # atmospheric temperature, degC
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
        time: np.ndarray,
        data: np.ndarray,
        window_minutes: float = 30,
        poly_degree: int = 2,
        min_points: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate interior gaps using polynomial curve fitting.

    Only fills gaps that have valid data on both sides (skips edges to avoid
    extrapolation). For each interior gap, fits a polynomial to data in a
    window before and after the gap, then uses the fitted curve to fill it.

    Returns:
        data_filled: array with interior gaps filled
        is_interpolated: boolean mask indicating which points were interpolated
    """
    missing = np.isnan(data)
    data_filled = data.copy()
    valid = ~missing

    if np.sum(valid) < 2:
        return data, missing

    window_hours = window_minutes / 60.0
    gaps = find_gaps(missing)
    is_interpolated = np.zeros_like(data, dtype=bool)

    for gap_start, gap_end in gaps:
        # only interpolate interior gaps (valid data on both sides)
        if not np.any(valid[:gap_start]) or not np.any(valid[gap_end:]):
            continue

        gap_time = time[gap_start:gap_end]

        # find valid data within window before and after gap
        before_mask = (time < time[gap_start]) & (time >= time[gap_start] - window_hours) & valid
        after_mask = (time > time[gap_end - 1]) & (time <= time[gap_end - 1] + window_hours) & valid

        fit_time = time[before_mask | after_mask]
        fit_data = data[before_mask | after_mask]

        if len(fit_data) < min_points:
            # fall back to linear interpolation
            idx_before = np.where((time < time[gap_start]) & valid)[0]
            idx_after = np.where((time > time[gap_end - 1]) & valid)[0]

            if len(idx_before) > 0 and len(idx_after) > 0:
                i0, i1 = idx_before[-1], idx_after[0]
                t0, t1 = time[i0], time[i1]
                y0, y1 = data[i0], data[i1]
                data_filled[gap_start:gap_end] = y0 + (y1 - y0) * (gap_time - t0) / (t1 - t0)
                is_interpolated[gap_start:gap_end] = True
            continue

        # fit polynomial (normalize time for numerical stability)
        t_mean = np.mean(fit_time)
        t_std = np.std(fit_time)
        if t_std < 1e-10:
            continue

        fit_time_norm = (fit_time - t_mean) / t_std
        gap_time_norm = (gap_time - t_mean) / t_std

        try:
            coeffs = np.polyfit(fit_time_norm, fit_data, poly_degree)
            data_filled[gap_start:gap_end] = np.polyval(coeffs, gap_time_norm)
            is_interpolated[gap_start:gap_end] = True
        except np.linalg.LinAlgError:
            continue

    return data_filled, is_interpolated


def compute_theta_850(flight: str, interpolate: bool = True) -> dict:
    filenames = MARLI_FILES[flight]

    all_time = []
    all_theta = []
    all_alt = []
    h_actual = None
    p_actual = None

    for filename in filenames:
        with open_dataset(DATASET, filename) as ds:
            time, theta, h, p = _extract_theta_850(ds)
            all_alt.append(ds[ALTITUDE].values)
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
        theta_850_filled, is_interpolated = interpolate_gaps(
            time_hours_regrid,
            theta_850_regrid,
            window_minutes=30,
            poly_degree=2,
            min_points=20,
        )
    else:
        theta_850_filled = theta_850_regrid
        is_interpolated = np.zeros_like(theta_850_regrid, dtype=bool)

    return {
        "time_utc_hours": time_hours_native,
        "theta_850": theta_850_filled,
        "time_regrid_utc_hours": time_hours_regrid,
        "altitude": altitude_native,
        "h_850": h_actual,
        "p_850": p_actual,
        "is_interpolated": is_interpolated,
    }


def plot_theta_850(
        flight: str,
        result: dict,
        theta_lim: tuple,
        alt_lim: tuple,
) -> str:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, f"{flight.lower()}_theta850.png")

    time_regrid = result["time_regrid_utc_hours"]
    theta = result["theta_850"]
    time_native = result["time_utc_hours"]
    alt = result["altitude"]
    is_interpolated = result["is_interpolated"]

    fig, ax = plt.subplots(figsize=(10, 5))

    # plot measured data with solid line
    measured_mask = ~is_interpolated
    ax.plot(
        time_regrid[measured_mask],
        theta[measured_mask],
        marker="o",
        markersize=3,
        linewidth=1,
        color="tab:blue",
        label=f"$\\theta_{850}${" (measured)" if np.any(is_interpolated) else ""}",
    )

    # plot interpolated data with dashed line
    if np.any(is_interpolated):
        ax.plot(
            time_regrid[is_interpolated],
            theta[is_interpolated],
            marker="o",
            markersize=3,
            linewidth=0,
            color="tab:green",
            alpha=0.6,
            label="$\\theta_{850}$ (interpolated)",
        )

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda h, _: f"{int(h):02d}:{int((h % 1) * 60):02d}")
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("$\\theta_{850}$ (K)")
    ax.set_ylim(theta_lim)

    ax2 = ax.twinx()
    ax2.plot(time_native, alt, color="black", linewidth=1.0, label="Aircraft altitude")
    ax2.set_ylabel("Aircraft Altitude (km)")
    ax2.set_ylim(alt_lim)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=9)

    ax.set_title(
        f"{flight}: Potential Temperature at ~850 hPa "
        f"(H={result['h_850']:.3f} km, p={result['p_850']:.1f} hPa)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def main():
    flights = list(MARLI_FILES.keys())

    # compute all results and find global ranges
    results = {}
    for flight in flights:
        results[flight] = compute_theta_850(flight, interpolate=True)

    all_theta = np.concatenate([r["theta_850"] for r in results.values()])
    all_alt = np.concatenate([r["altitude"] for r in results.values()])
    theta_lim = (np.nanmin(all_theta), np.nanmax(all_theta))
    alt_lim = (np.nanmin(all_alt), np.nanmax(all_alt))

    # plot with fixed ranges
    for flight in flights:
        result = results[flight]
        measured = np.count_nonzero(~result["is_interpolated"])
        interpolated = np.count_nonzero(result["is_interpolated"])
        total = len(result["theta_850"])
        plot = plot_theta_850(flight, result, theta_lim, alt_lim)
        print(
            f"{flight}: theta_850 at H={result['h_850']:.4f} km "
            f"(p={result['p_850']:.1f} hPa), "
            f"{measured} measured, {interpolated} interpolated ({total} total) -> {plot}"
        )


if __name__ == "__main__":
    main()
