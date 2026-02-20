import os
from dataclasses import dataclass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from nc.loader import DATASET_VARS, PROJECT_ROOT, open_dataset
from nc.segmentation import find_inflection_points

_vars = DATASET_VARS["638-038"]
ALTITUDE = _vars["altitude"]
TIME = _vars["time"]
DATASET = "638-038"
PLOTS_DIR = os.path.join(PROJECT_ROOT, f"output/{DATASET}/plots/segments")

# per-flight RDP epsilon values (tuned visually)
EPSILONS = {
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


@dataclass
class FlightSegments:
    """Inflection points and associated data for a single flight."""

    flight: str
    times: np.ndarray  # datetime64 array, full resolution
    altitude: np.ndarray  # km, full resolution
    inflection_indices: np.ndarray  # indices into times/altitude

    @property
    def n_points(self) -> int:
        return len(self.inflection_indices)

    @property
    def n_segments(self) -> int:
        return self.n_points - 1

    def segment_slice(self, start_pt: int, end_pt: int) -> slice:
        """Get a slice between two inflection point indices."""
        return slice(self.inflection_indices[start_pt], self.inflection_indices[end_pt])

    def segment_times(self, start_pt: int, end_pt: int) -> np.ndarray:
        return self.times[self.segment_slice(start_pt, end_pt)]

    def segment_altitude(self, start_pt: int, end_pt: int) -> np.ndarray:
        return self.altitude[self.segment_slice(start_pt, end_pt)]

    def point_time(self, pt: int) -> np.datetime64:
        return self.times[self.inflection_indices[pt]]

    def point_altitude(self, pt: int) -> float:
        return self.altitude[self.inflection_indices[pt]]


def load_flight_segments(flight: str, epsilon: float = None) -> FlightSegments:
    """Load flight data and compute RDP inflection points.

    Args:
        flight: Flight identifier (e.g., 'RF10').
        epsilon: RDP tolerance (km). If None, uses the tuned value from EPSILONS.
    """
    if epsilon is None:
        epsilon = EPSILONS[flight]

    with open_dataset(DATASET, flight) as ds:
        times = ds[TIME].values
        altitude = ds[ALTITUDE].values / 1000.0  # m to km

    mask = find_inflection_points(altitude, epsilon=epsilon)
    inflection_indices = np.where(mask)[0]

    return FlightSegments(
        flight=flight,
        times=times,
        altitude=altitude,
        inflection_indices=inflection_indices,
    )


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


if __name__ == "__main__":
    for flight in EPSILONS:
        fs = load_flight_segments(flight)
        plot_flight_segments(fs)
