"""Spatial data loaders for diagnostic plotting.

Delegates expensive data loading to the cached _load_model_arrays from
features.py; load_diagnostic_spatial itself is not cached so adding
variables here never invalidates the underlying model data cache.
"""

import numpy as np
import pandas as pd
import xarray as xr

from nc.cache import MEMORY
from nc.loader import open_file
from nc.remote import SWING3_MODELS
from nc.vars import SWING3 as v
from nc.vars import SWING3_LMDZ as v_lmdz
from swing3.features import _load_model_arrays
from swing3.grids import crop_region

DIAG_LABELS: dict[str, str] = {
    "pe": "Precipitation efficiency (%)",
    "mcao": "MCAO (K)",
    "ivt": "IVT (kg m^{-1} s^{-1})",
    "wind_sfc": "Wind 925 hPa (m/s)",
    "ts": "Temperature (surface) (deg C)",
    "sh": "Specific humidity (surface) (kg/kg)",
    "qvsum": "Atmospheric column precipitable water (kg/m^2)",
    "q_700": "Specific humidity (700 hPa) (kg/kg)",
    "t_700": "Temperature (700 hPa) (deg C)",
    "omega_925": "omega 925 hPa (Pa/s)",
    "omega_700": "omega 700 hPa (Pa/s)",
    "dD_gradient": "dD Gradient (per mil)",
    "dDp": "Delta of precipitation, dD (per mil)",
    "dexcessp": "Delta of precipitation, d-excess (per mil)",
    "dDs": "Delta of specific humidity (surface), dD (per mil)",
    "dexcesss": "Delta of specific humidity (surface), d-excess (per mil)",
    "low_cloud": "Low cloud fraction (%)",
}


@MEMORY.cache
def _model_coords(model: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) arrays cropped to CAESAR_BOUNDS. Lightweight metadata read."""
    raw_time_dim = v_lmdz.time if model == "LMDZ" else v.time
    with open_file(SWING3_MODELS[model], decode_times=False) as ds:
        sample = crop_region(ds[v.precip_efficiency].isel({raw_time_dim: 0})).load()
    return sample["lat"].values, sample["lon"].values


def load_diagnostic_spatial(model: str) -> dict[str, xr.DataArray]:
    """Load key variables as (time, lat, lon) DataArrays cropped to CAESAR_BOUNDS.

    Variables: pe, mcao, ivt, wind_sfc, ts, sh, qvsum, q_700, t_700,
    omega_925, omega_700, dD_gradient, dDp, dexcessp, dDs, dexcesss,
    and low_cloud (only for models that have cloud data).
    """
    arrays, time_groups = _load_model_arrays(model)
    lat_vals, lon_vals = _model_coords(model)

    n_jfma = int(time_groups.max()) + 1
    nlat = len(lat_vals)
    nlon = len(lon_vals)

    jfma_dates = pd.DatetimeIndex(
        [
            pd.Timestamp(year=1979 + i // 4, month=[1, 2, 3, 4][i % 4], day=1)
            for i in range(n_jfma)
        ]
    )

    def _to_da(key: str) -> xr.DataArray:
        return xr.DataArray(
            arrays[key].reshape(n_jfma, nlat, nlon),
            dims=["time", "lat", "lon"],
            coords={"time": jfma_dates, "lat": lat_vals, "lon": lon_vals},
        )

    result: dict[str, xr.DataArray] = {
        "pe": _to_da("pref"),
        "mcao": _to_da("mcao"),
        "ivt": _to_da("ivt"),
        "wind_sfc": _to_da("wind_sfc"),
        "ts": _to_da("ts"),
        "sh": _to_da("sh"),
        "qvsum": _to_da("qvsum"),
        "q_700": _to_da("q_700"),
        "t_700": _to_da("t_700"),
        "omega_925": _to_da("omega_925"),
        "omega_700": _to_da("omega_700"),
        "dD_gradient": _to_da("dD_gradient"),
        "dDp": _to_da("dDp"),
        "dexcessp": _to_da("dexcessp"),
        "dDs": _to_da("dDs"),
        "dexcesss": _to_da("dexcesss"),
    }

    lc_da = _to_da("low_cloud")
    if not np.isnan(lc_da.values).all():
        result["low_cloud"] = lc_da

    return result
