import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import xarray as xr
from metpy.units import units as munits

from nc.cache import MEMORY
from nc.flights import CAESAR_BOUNDS as bounds
from nc.loader import open_file
from nc.remote import SWING3_MODELS
from nc.vars import SWING3 as v
from nc.vars import SWING3_LMDZ as v_lmdz
from nc.vars import SWING3_SST as v_sst
from swing3.sst import load_sst

P_850 = 850  # hPa


def jfma_indices(n_times: int) -> np.ndarray:
    """Indices of Jan–Apr months in a monthly series of length n_times starting 1979-01-01."""
    JFMA = (1, 2, 3, 4)  # January through April

    dates = pd.date_range("1979-01-01", periods=n_times, freq="MS")
    return np.where(dates.month.isin(JFMA))[0]


@MEMORY.cache
def _load_model_data(
    model: str, sst_da: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, str]:
    """
    Load Jan–Apr MCAO (SST − θ850) and PE for one model, cropped to CAESAR_BOUNDS.

    Returns (mcao, pref, time_dim) as DataArrays of shape (time, lat, lon).
    """

    def _sel_region(da):
        """Filter CAESAR bounds and correct longitude (degrees East)"""
        da = da.assign_coords(lon=((da.lon + 180) % 360 - 180)).sortby("lon")
        return da.sortby("lat").sel(
            lat=slice(bounds["MIN_LAT"], bounds["MAX_LAT"]),
            lon=slice(bounds["MIN_LON"], bounds["MAX_LON"]),
        )

    time_dim = v_lmdz.time if model == "LMDZ" else v.time
    n_sst = sst_da.sizes[v_sst.time]
    sst_region = _sel_region(sst_da)

    with open_file(SWING3_MODELS[model], decode_times=False) as ds:
        n_times = ds.sizes[time_dim]
        n_min = min(n_times, n_sst)

        if n_min < max(n_times, n_sst):
            print(
                f"\t[{model}] time mismatch: model={n_times}, SST={n_sst}; using {n_min}"
            )

        jfma = jfma_indices(n_min)

        t850 = _sel_region(
            ds[v.temperature]
            .sel({v.pressure: P_850}, method="nearest")
            .isel({time_dim: jfma})
        ).load()
        theta850 = (
            mpcalc.potential_temperature(
                P_850 * munits.hPa,
                t850.values * munits.degC,
            )
            .to("K")
            .magnitude
        )

        pref = _sel_region(ds[v.precip_efficiency].isel({time_dim: jfma})).load()

    sst_jfma = sst_region.values[jfma]

    mcao = pref.copy(data=sst_jfma - theta850)
    return mcao, pref, time_dim


def load_mcao_pe(
    model: str, sst_da: xr.DataArray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    1-D arrays of valid MCAO and PE over Jan–Apr x lat x lon.

    Pass sst_da to reuse an already-loaded SST DataArray across models.
    """
    if sst_da is None:
        sst_da = load_sst()

    mcao, pref, _ = _load_model_data(model, sst_da)

    mcao_flat = mcao.values.ravel()
    pe_flat = pref.values.ravel()
    mask = np.isfinite(mcao_flat) & np.isfinite(pe_flat)
    return mcao_flat[mask], pe_flat[mask]


def load_mcao_pe_clim(
    model: str, sst_da: xr.DataArray | None = None
) -> tuple[xr.DataArray, xr.DataArray]:
    """Jan–Apr climatological mean MCAO and PE on the model's native grid (lat x lon)."""
    if sst_da is None:
        sst_da = load_sst()

    mcao, pref, time_dim = _load_model_data(model, sst_da)
    return mcao.mean(dim=time_dim), pref.mean(dim=time_dim)
