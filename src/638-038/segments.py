import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from nc.loader import PROJECT_ROOT, load_dataset
from nc.segmentation import find_inflection_points

ALTITUDE = "alt"
TIME = "time"
DATASET = "638-038"
PLOTS_DIR = os.path.join(PROJECT_ROOT, f"output/{DATASET}/plots/segments")
os.makedirs(PLOTS_DIR, exist_ok=True)

if __name__ == "__main__":
    FLIGHTS = {
        "RF01": 0.25,
        "RF02": 0.1,
        "RF03": 0.05,
        "RF04": 0.25,
        "RF05": 0.25,
        "RF06": 0.1,
        "RF07": 0.5,
        "RF09": 0.1,
        "RF10": 0.1,
    }

    for flight, epsilon in FLIGHTS.items():
        ds = load_dataset(DATASET, flight)
        times = ds[TIME].values
        alt = ds[ALTITUDE].values / 1000.0
        ds.close()

        mask = find_inflection_points(alt, epsilon=epsilon)
        inflection_idx = np.where(mask)[0]
        colors = ["tab:blue", "tab:orange"]

        fig, ax_idx = plt.subplots(figsize=(15, 5))

        # altitude with inflection points
        ax_idx.plot(range(len(alt)), alt, color="black", linewidth=1)
        ax_idx.scatter(inflection_idx, alt[mask], color="red", s=10, zorder=5)

        # color segments
        for i in range(len(inflection_idx) - 1):
            start = inflection_idx[i]
            end = inflection_idx[i + 1]
            ax_idx.axvspan(start, end, color=colors[i % 2], alpha=0.2)

        # inflection point indices
        ax_idx.set_xticks(inflection_idx)
        ax_time = ax_idx.twiny()
        labels = [str(i) if i % 5 == 0 else "" for i in range(len(inflection_idx))]
        ax_idx.set_xticklabels(labels, fontsize=6)
        ax_idx.set_xlabel("Inflection Point Index")
        ax_idx.set_xlim(0, len(alt) - 1)

        # UTC time
        ax_time.set_xlim(mdates.date2num(times[0]), mdates.date2num(times[-1]))
        ax_time.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax_time.set_xlabel("Time (UTC)")

        ax_idx.set_ylabel("Altitude (km)")
        ax_idx.set_title(f"{flight} segmentation ($\\epsilon$={epsilon})", pad=30)
        ax_idx.grid(True, alpha=0.2)

        out_path = os.path.join(PLOTS_DIR, f"{flight.lower()}_segments.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
