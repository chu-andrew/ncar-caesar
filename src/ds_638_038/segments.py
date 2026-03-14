from dataclasses import dataclass

import numpy as np
from rdp import rdp

from nc.loader import open_dataset
from nc.vars import DS_638_038 as v

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


def find_inflection_points(series: np.ndarray, epsilon: float) -> np.ndarray:
    points = np.column_stack([np.arange(len(series)), series])
    return rdp(points, epsilon=epsilon, return_mask=True)


def load_flight_segments(flight: str, epsilon: float = None) -> FlightSegments:
    """Load flight data and compute RDP inflection points.

    Args:
        flight: Flight identifier (e.g., 'RF10').
        epsilon: RDP tolerance (km). If None, uses the tuned value from EPSILONS.
    """
    if epsilon is None:
        epsilon = EPSILONS[flight]

    with open_dataset(v.dataset, flight) as ds:
        times = ds[v.time].values
        altitude = ds[v.altitude].values / 1000.0  # m to km

    mask = find_inflection_points(altitude, epsilon=epsilon)
    inflection_indices = np.where(mask)[0]

    return FlightSegments(
        flight=flight,
        times=times,
        altitude=altitude,
        inflection_indices=inflection_indices,
    )
