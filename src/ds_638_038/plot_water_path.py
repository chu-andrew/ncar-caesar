import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns

from nc.flights import LOW_LEVEL_LEGS
from nc.loader import PROJECT_ROOT, open_dataset
from ds_638_038.segments import load_flight_segments

DATASET = "638-038"
TIME = "time"
ALTITUDE = "alt"
LWP = "LWP"
WVP = "WVP"

PLOTS_DIR = os.path.join(PROJECT_ROOT, f"output/{DATASET}/plots/water_path")


def plot_water_path(flight: str, start_pt: int, end_pt: int):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fs = load_flight_segments(flight)

    with open_dataset(DATASET, flight) as ds:
        times = ds[TIME].values
        lwp = ds[LWP].values
        wvp = ds[WVP].values
        alt = ds[ALTITUDE].values

    s = fs.segment_slice(start_pt, end_pt)
    times, lwp, wvp, alt = times[s], lwp[s], wvp[s], alt[s]
    title = (
        f"{flight} Liquid Water Path & Water Vapor Path (points {start_pt}-{end_pt})"
    )

    if len(times) == 0:
        print(f"No data for selected range in {flight}.")
        return

    fig = plt.figure(figsize=(15, 12))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)

    sns.lineplot(x=times, y=lwp, ax=ax1, linewidth=1.0)
    ax1.set_ylabel("LWP $(g/m^2)$")
    ax1.set_title(title)

    sns.lineplot(x=times, y=wvp, ax=ax2, linewidth=1.0)
    ax2.set_ylabel("WVP $(g/m^2)$")

    sns.lineplot(x=times, y=alt, ax=ax3, linewidth=1.0)
    ax3.set_ylabel("Altitude (m)")
    ax3.set_xlabel("Time (UTC)")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    out_path = os.path.join(
        PLOTS_DIR, f"{flight.lower()}_pt{start_pt:02}-{end_pt:02}.png"
    )
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    for flight in LOW_LEVEL_LEGS:
        for start, end in LOW_LEVEL_LEGS[flight]:
            plot_water_path(flight, start, end)


if __name__ == "__main__":
    main()
