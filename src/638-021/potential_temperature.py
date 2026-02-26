"""
Calculate potential temperature at 850 hPa.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from nc.flights import FLIGHTS, MARLI_FILES
from nc.loader import DATASET_VARS, PROJECT_ROOT, open_dataset
from nc.time import utc_hours_to_datetime64

_vars = DATASET_VARS["638-021"]
ALTITUDE = _vars["altitude"]

DATASET = "638-021"
PLOTS_DIR = os.path.join(PROJECT_ROOT, f"output/{DATASET}/plots/potential_temperature")

FILL_VALUE = 9999.0
P_850 = 850  # hPa

MAD_K = 5.0  # multiplier for MAD


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


def compute_theta_850(flight: str) -> dict:
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

    time_hours = np.concatenate(all_time)
    theta_850 = np.concatenate(all_theta)
    altitude = np.concatenate(all_alt)

    date_str = FLIGHTS[flight]
    time_dt = utc_hours_to_datetime64(time_hours, date_str)

    return {
        "time": time_dt,
        "time_utc_hours": time_hours,
        "theta_850": theta_850,
        "altitude": altitude,
        "h_850": h_actual,
        "p_850": p_actual,
    }


def plot_theta_850(flight: str, result: dict) -> str:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, f"{flight.lower()}_theta850.png")

    time = result["time_utc_hours"]
    theta = result["theta_850"]
    alt = result["altitude"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, theta, marker="o", markersize=3, linewidth=1, label="$\\theta_{850}$")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda h, _: f"{int(h):02d}:{int((h % 1) * 60):02d}")
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("$\\theta_{850}$ (K)")

    ax2 = ax.twinx()
    ax2.plot(time, alt, color="black", linewidth=1.5, label="Aircraft altitude")
    ax2.set_ylabel("Aircraft Altitude (km)")

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
    flights = MARLI_FILES.keys()

    for flight in flights:
        result = compute_theta_850(flight)
        valid = np.count_nonzero(~np.isnan(result["theta_850"]))
        total = len(result["theta_850"])
        plot = plot_theta_850(flight, result)
        print(
            f"{flight}: theta_850 at H={result['h_850']:.4f} km "
            f"(p={result['p_850']:.1f} hPa), "
            f"{valid}/{total} valid profiles -> {plot}"
        )


if __name__ == "__main__":
    main()
