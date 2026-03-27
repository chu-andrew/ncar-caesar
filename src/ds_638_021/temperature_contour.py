import numpy as np

from nc.cache import MEMORY
from nc.flights import MARLI_FILES
from nc.loader import open_dataset
from nc.vars import DS_638_021 as v


def mask_temperature_outliers(T: np.ndarray) -> np.ndarray:
    """
    Mask outlier temperatures using per-level median absolute deviation.
    """

    MAD_MULT = 5.0  # multiplier for MAD

    T = T.astype(np.float64, copy=True)
    T[(T >= v.fill_value) | (T < -100) | (T > 100)] = np.nan

    if T.ndim == 1:
        T = T[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    for j in range(T.shape[1]):
        col = T[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) < 3:
            continue
        median = np.median(valid)
        mad = np.median(np.abs(valid - median))
        if mad == 0:
            continue
        T[np.abs(col - median) > MAD_MULT * mad, j] = np.nan

    return T[:, 0] if squeeze else T


@MEMORY.cache
def load_contour_data(flight: str) -> dict:
    filenames = MARLI_FILES[flight]

    # use the first file's height grid as reference (RF07 has two datasets)
    with open_dataset(v.dataset, filenames[0]) as ds:
        H = ds["H"].values

    all_time = []
    all_T = []
    all_alt = []

    for filename in filenames:
        with open_dataset(v.dataset, filename) as ds:
            h_file = ds["H"].values
            t_data = ds["T"].values.astype(np.float64)
            all_time.append(ds[v.time].values)
            all_alt.append(ds[v.altitude].values)

            if h_file.shape[0] == H.shape[0]:
                all_T.append(t_data)
            else:
                # interpolate from file's height grid (h_file) to reference grid (H)
                all_T.append(
                    np.array(
                        [
                            np.interp(H, h_file, t_data[i])
                            for i in range(t_data.shape[0])
                        ]
                    )
                )

    T = mask_temperature_outliers(np.concatenate(all_T, axis=0))

    return {
        "H": H,
        "time": np.concatenate(all_time),
        "T": T,
        "alt": np.concatenate(all_alt),
    }
