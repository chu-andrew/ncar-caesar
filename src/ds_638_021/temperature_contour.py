import numpy as np

from nc.flights import MARLI_FILES
from nc.loader import open_dataset
from nc.vars import DS_638_021 as v

from ds_638_021.potential_temperature import (
    P_850,
    find_nearest_pressure_bin,
    height_to_pressure,
    mask_temperature_outliers,
)
from nc.units import celsius_to_kelvin, wvmr_to_specific_humidity


def load_contour_data(flight: str) -> dict:
    filenames = MARLI_FILES[flight]

    # use the first file's height grid as reference (RF07 has two datasets)
    with open_dataset(v.dataset, filenames[0]) as ds:
        H = ds["H"].values

    all_time = []
    all_T = []
    all_WVMR = []
    all_alt = []

    for filename in filenames:
        with open_dataset(v.dataset, filename) as ds:
            h_file = ds["H"].values
            t_data = ds["T"].values.astype(np.float64)
            wvmr_data = ds[v.wvmr].values.astype(np.float64)
            all_time.append(ds[v.time].values)
            all_alt.append(ds[v.altitude].values)

            if h_file.shape[0] == H.shape[0]:
                all_T.append(t_data)
                all_WVMR.append(wvmr_data)
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
                all_WVMR.append(
                    np.array(
                        [
                            np.interp(H, h_file, wvmr_data[i])
                            for i in range(wvmr_data.shape[0])
                        ]
                    )
                )

    T = mask_temperature_outliers(np.concatenate(all_T, axis=0))
    WVMR = np.concatenate(all_WVMR, axis=0)
    WVMR[WVMR >= v.fill_value] = np.nan

    # precompute 850 hPa height using virtual temperature
    p_levels = height_to_pressure(
        H, celsius_to_kelvin(T), wvmr_to_specific_humidity(WVMR)
    )
    idx_850, _ = find_nearest_pressure_bin(p_levels, P_850, H)

    return {
        "H": H,
        "time": np.concatenate(all_time),
        "T": T,
        "WVMR": WVMR,
        "alt": np.concatenate(all_alt),
        "h_850": float(H[idx_850]),
    }
