import os

import matplotlib.pyplot as plt
import numpy as np

from ds_638_021.potential_temperature import P_850, height_to_pressure
from ds_638_021.temperature_contour import load_contour_data
from ds_638_052.cloud_base import load_cloud_base
from nc.flights import MARLI_FILES
from nc.loader import PROJECT_ROOT
from nc.vars import DS_638_021 as v

PLOTS_DIR_TC = os.path.join(
    PROJECT_ROOT, f"output/{v.dataset}/plots/temperature_contour"
)


def plot_temperature_contour(flight: str, data: dict, vmin: float, vmax: float) -> str:
    H = data["H"]
    time = data["time"]
    T = data["T"]
    alt = data["alt"]

    fig, ax = plt.subplots(figsize=(12, 6))
    cf = ax.pcolormesh(
        time, H, T.T, shading="auto", cmap="RdYlBu_r", vmin=vmin, vmax=vmax
    )
    fig.colorbar(cf, ax=ax, label="Temperature (°C)")

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda h, _: f"{int(h):02d}:{int((h % 1) * 60):02d}")
    )

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

    cb_time, cb_height = load_cloud_base(flight)
    cb_height_km = cb_height / 1000.0
    valid = ~np.isnan(cb_height_km)
    ax.scatter(
        cb_time[valid],
        cb_height_km[valid],
        s=1,
        color="tab:green",
        alpha=0.4,
        label="Cloud base (WCL)",
    )

    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Height (km)")
    ax.set_title(f"{flight}: MARLi Atmospheric Temperature (°C)")

    os.makedirs(PLOTS_DIR_TC, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR_TC, f"{flight.lower()}_temperature_contour.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    flights = list(MARLI_FILES.keys())

    all_data = {}
    for flight in flights:
        all_data[flight] = load_contour_data(flight)

    vmin = min(np.nanmin(d["T"]) for d in all_data.values())
    vmax = max(np.nanmax(d["T"]) for d in all_data.values())

    for flight in flights:
        path = plot_temperature_contour(flight, all_data[flight], vmin, vmax)
        print(f"{flight} -> {path}")


if __name__ == "__main__":
    main()
