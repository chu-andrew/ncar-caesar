import numpy as np

from nc.units import S_PER_HR, NS_PER_S


def utc_hours_to_datetime64(hours: np.ndarray, date_str: str) -> np.ndarray:
    """
    Convert UTC-hours float array to datetime64[ns] given a date string.

    Args:
        hours: Array of UTC hours (e.g., 12.5 = 12:30:00).
        date_str: Date string in 'YYYY-MM-DD' format.

    Returns:
        Array of numpy datetime64[ns] values.
    """
    base = np.datetime64(date_str, "ns")
    seconds = (hours * S_PER_HR).astype(np.float64)
    offsets = (seconds * NS_PER_S).astype("timedelta64[ns]")
    return base + offsets


def seconds_to_datetime64(seconds: np.ndarray, date_str: str) -> np.ndarray:
    """Convert seconds-from-midnight to datetime64[ns] given a date string."""
    base = np.datetime64(date_str, "ns")
    ns_per_sec = np.int64(NS_PER_S)
    return base + (seconds * ns_per_sec).astype("timedelta64[ns]")
