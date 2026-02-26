import os

import matplotlib.pyplot as plt
import numpy as np

from nc.flights import MARLI_FILES
from nc.loader import DATASET_VARS, PROJECT_ROOT, open_dataset
from potential_temperature import height_to_pressure, mask_temperature_outliers, P_850

DATASET = "638-021"
PLOTS_DIR = os.path.join(PROJECT_ROOT, f"output/{DATASET}/plots/temperature_contour")
_vars = DATASET_VARS["638-021"]
ALTITUDE = _vars["altitude"]
TIME = _vars["time"]


def plot_temperature_contour(flight: str) -> str:
    filenames = MARLI_FILES[flight]

    # use the first file's height grid as reference
    with open_dataset(DATASET, filenames[0]) as ds:
        H = ds["H"].values

    all_time = []
    all_T = []
    all_alt = []

    for filename in filenames:
        with open_dataset(DATASET, filename) as ds:
            h_file = ds["H"].values
            t_data = ds["T"].values.astype(np.float64)
            all_time.append(ds[TIME].values)
            all_alt.append(ds[ALTITUDE].values)

            if h_file.shape[0] == H.shape[0]:
                all_T.append(t_data)
            else:
                # interpolate onto the reference H grid
                t_interp = np.array(
                    [np.interp(H, h_file, t_data[i]) for i in range(t_data.shape[0])]
                )
                all_T.append(t_interp)

    time = np.concatenate(all_time)
    T = np.concatenate(all_T, axis=0)
    alt = np.concatenate(all_alt)

    # mask fill values and per-level MAD outliers
    T = mask_temperature_outliers(T)

    fig, ax = plt.subplots(figsize=(12, 6))
    cf = ax.pcolormesh(time, H, T.T, shading="auto", cmap="RdYlBu_r")
    fig.colorbar(cf, ax=ax, label="Temperature (°C)")

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda h, _: f"{int(h):02d}:{int((h % 1) * 60):02d}")
    )

    # horizontal line at the height closest to 850 hPa
    p_levels = height_to_pressure(H)
    idx_850 = np.argmin(np.abs(p_levels - P_850))
    h_850 = H[idx_850]
    ax.axhline(
        h_850,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"~850 hPa (H={h_850:.3f} km)",
    )

    ax.plot(time, alt, color="black", linewidth=1.5, label="Aircraft altitude")

    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Height (km)")
    ax.set_title(f"{flight}: MARLi Atmospheric Temperature (°C)")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, f"{flight.lower()}_temperature_contour.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    for flight in MARLI_FILES:
        path = plot_temperature_contour(flight)
        print(f"{flight} -> {path}")


if __name__ == "__main__":
    main()
