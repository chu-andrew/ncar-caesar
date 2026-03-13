import numpy as np

from nc.flights import MARLI_FILES
from nc.loader import open_dataset
from nc.vars import DS_638_021 as v

from ds_638_021.potential_temperature import mask_temperature_outliers


def load_contour_data(flight: str) -> dict:
    filenames = MARLI_FILES[flight]

    # use the first file's height grid as reference
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
                t_interp = np.array(
                    [np.interp(H, h_file, t_data[i]) for i in range(t_data.shape[0])]
                )
                all_T.append(t_interp)

    T = mask_temperature_outliers(np.concatenate(all_T, axis=0))

    return {
        "H": H,
        "time": np.concatenate(all_time),
        "T": T,
        "alt": np.concatenate(all_alt),
    }
