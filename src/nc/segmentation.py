import numpy as np
from rdp import rdp


def find_inflection_points(series: np.ndarray, epsilon: float) -> np.ndarray:
    points = np.column_stack([np.arange(len(series)), series])
    return rdp(points, epsilon=epsilon, return_mask=True)
