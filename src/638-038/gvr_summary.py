import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from nc.flights import FLIGHTS
from nc.loader import DATASET_VARS, PROJECT_ROOT, open_dataset

DATASET = "638-038"
PLOTS_DIR = os.path.join(PROJECT_ROOT, f"output/{DATASET}/plots/gvr_summary")
_vars = DATASET_VARS[DATASET]
TIME = _vars["time"]
ALTITUDE = _vars["altitude"]


def plot_gvr_summary(flight: str) -> str:
    with open_dataset(DATASET, flight) as ds:
        times = ds[TIME].values
        alt = ds[ALTITUDE].values / 1000.0  # m -> km
        lwp = ds["LWP"].values
        wvp = ds["WVP"].values

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax1.plot(times, alt, color="black", linewidth=1.0)
    ax1.set_ylabel("Altitude (km)")
    ax1.set_title(f"{flight}: GVR Summary")

    ax2.plot(times, lwp, color="tab:blue", linewidth=0.8)
    ax2.set_ylabel("LWP ($g/m^2$)")

    ax3.plot(times, wvp, color="tab:green", linewidth=0.8)
    ax3.set_ylabel("WVP ($g/m^2$)")
    ax3.set_xlabel("Time (UTC)")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, f"{flight.lower()}_gvr_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    for flight in FLIGHTS:
        path = plot_gvr_summary(flight)
        print(f"{flight} -> {path}")


if __name__ == "__main__":
    main()
