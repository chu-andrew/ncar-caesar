import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from nc.loader import PROJECT_ROOT
from ds_638_038.segments import load_flight_segments, FlightSegments, EPSILONS

DATASET = "638-038"
PLOTS_DIR = os.path.join(PROJECT_ROOT, f"output/{DATASET}/plots/segments")


def plot_flight_segments(fs: FlightSegments):
    """Plot altitude with RDP inflection points and alternating segment colors."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, ax_idx = plt.subplots(figsize=(12, 4))

    # altitude with inflection points
    ax_idx.plot(range(len(fs.altitude)), fs.altitude, color="black", linewidth=1)
    ax_idx.scatter(
        fs.inflection_indices,
        fs.altitude[fs.inflection_indices],
        color="red",
        s=10,
        zorder=5,
    )

    # alternating segment colors
    colors = ["tab:blue", "tab:orange"]
    for i in range(fs.n_segments):
        start = fs.inflection_indices[i]
        end = fs.inflection_indices[i + 1]
        ax_idx.axvspan(start, end, color=colors[i % 2], alpha=0.2)

    # inflection point indices
    ax_idx.set_xticks(fs.inflection_indices)
    labels = [str(i) if i % 5 == 0 else "" for i in range(fs.n_points)]
    ax_idx.set_xticklabels(labels, fontsize=6)
    ax_idx.set_xlabel("Inflection Point Index")
    ax_idx.set_xlim(0, len(fs.altitude) - 1)

    # UTC time
    ax_time = ax_idx.twiny()
    ax_time.set_xlim(mdates.date2num(fs.times[0]), mdates.date2num(fs.times[-1]))
    ax_time.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_time.set_xlabel("Time (UTC)")

    ax_idx.set_ylabel("Altitude (km)")
    epsilon = EPSILONS.get(fs.flight, "?")
    ax_idx.set_title(f"{fs.flight} segmentation ($\\epsilon$={epsilon})", pad=30)
    ax_idx.grid(True, alpha=0.2)

    out_path = os.path.join(PLOTS_DIR, f"{fs.flight.lower()}_segments.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    for flight in EPSILONS:
        fs = load_flight_segments(flight)
        plot_flight_segments(fs)


if __name__ == "__main__":
    main()
