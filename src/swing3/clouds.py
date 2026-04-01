from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from nc.cache import MEMORY
from nc.loader import open_file
from nc.remote import CLOUD_DIR, SWING3_MODELS
from nc.vars import SWING3 as v
from nc.vars import SWING3_LMDZ as v_lmdz
from swing3.models import _crop_region, JFMA, START_YEAR, END_YEAR

MODELS = list(SWING3_MODELS.keys())


class CloudSpec(NamedTuple):
    glob: str
    cloud_cover_var: str
    scale: float  # multiply to get percent (0-100)
    pressure_slice: tuple[int, int] | None  # (lo, hi) hPa


CLOUD_VAR_MAP: dict[str, CloudSpec] = {
    "CAM6": CloudSpec("*CLDLOW*", "CLDLOW", 100.0, None),
    "ECHAM": CloudSpec("*low_cld*", "low_cld", 100.0, None),
    "GISS": CloudSpec("*CloudCover*", "pcldl", 1.0, None),
    "GSM": CloudSpec("*CloudCover*", "tcdclcl", 1.0, None),
    "LMDZ": CloudSpec("*CloudCover*", "cldl", 100.0, None),
    "MIROC": CloudSpec(
        "*CloudCover*", "cldfrc", 1.0, (850, 1000)
    ),  # TODO: verify low-cloud definition
}


@MEMORY.cache
def load_low_cloud(model: str) -> xr.DataArray:
    """
    Load JFMA low-cloud fraction for the given model, 1979–2023, on its native grid.

    Returns an xr.DataArray (time, lat, lon) in percent (0-100) cropped to
    CAESAR_BOUNDS.  Returns an all-NaN DataArray if cloud data is unavailable.
    """
    spec = CLOUD_VAR_MAP.get(model)
    if spec is None:
        print(f"\t[{model}] no cloud spec, skipping")
        return _nan_da()

    cloud_path = CLOUD_DIR / ("ECHAM6" if model == "ECHAM" else model)
    files = sorted(cloud_path.glob(spec.glob))

    assert len(files) == 1
    cloud_file = files[0]
    time_dim = v_lmdz.time if model == "LMDZ" else v.time
    open_kwargs = (
        {"decode_times": False}
        if model in {"GISS", "GSM", "MIROC"}
        else {"decode_times": xr.coders.CFDatetimeCoder(use_cftime=True)}
    )

    with open_file(cloud_file, **open_kwargs) as ds:
        ds = _fix_time(ds, model, time_dim)
        da = ds[spec.cloud_cover_var]
        da = _crop_region(da)

        if spec.pressure_slice is not None:
            p_lo, p_hi = spec.pressure_slice
            da = da.sel({v.pressure: slice(p_hi, p_lo)}).mean(
                dim=v.pressure
            )  # TODO: verify methodology for MIROC cloud fraction calculations
        if model == "LMDZ":
            # resample time from daily to monthly
            da = da.resample(
                {time_dim: "MS"}
            ).mean()  # TODO verify resampling methodology
        da = _select_jfma(da, time_dim).load()

    result = da * spec.scale
    result.attrs["units"] = "%"
    result.attrs["long_name"] = "Low cloud fraction"
    return result


def load_low_cloud_clim(model: str) -> xr.DataArray:
    """JFMA climatological mean low-cloud fraction on the native grid."""
    da = load_low_cloud(model)
    time_dim = v_lmdz.time if model == "LMDZ" else v.time
    if time_dim in da.dims:
        return da.mean(dim=time_dim)
    return da


def load_low_cloud_t42(model: str) -> xr.DataArray:
    """JFMA low-cloud fraction time series regridded to the T42 grid."""
    da = load_low_cloud(model)
    lat, lon = _t42_grid()

    # FIXME: consider using xesmf for proper regridding
    return da.interp(lat=lat, lon=lon, method="nearest")


def load_low_cloud_clim_t42(model: str) -> xr.DataArray:
    """JFMA climatological mean low-cloud fraction regridded to the T42 grid."""
    time_dim = v_lmdz.time if model == "LMDZ" else v.time
    return load_low_cloud_t42(model).mean(dim=time_dim)


@MEMORY.cache
def _t42_grid() -> tuple[np.ndarray, np.ndarray]:
    """T42 grid lat/lon from a reference SWING3 model, cropped to CAESAR_BOUNDS."""
    ref_model = MODELS[0]
    time_dim = v_lmdz.time if ref_model == "LMDZ" else v.time
    with open_file(SWING3_MODELS[ref_model], decode_times=False) as ds:
        ref = _crop_region(ds[v.precip_efficiency].isel({time_dim: 0}))
        return ref.lat.values, ref.lon.values


def _fix_time(ds: xr.Dataset, model: str, time_dim: str) -> xr.Dataset:
    if model == "GISS":
        # "day as %Y%m%d.%f"; e.g. 19790115.0
        vals = ds[time_dim].values.astype(int)
        years, months, days = vals // 10000, (vals // 100) % 100, vals % 100
        dates = pd.to_datetime(
            [f"{y:04d}-{m:02d}-{d:02d}" for y, m, d in zip(years, months, days)]
        )
        ds = ds.assign_coords({time_dim: dates})
    elif model in {"GSM", "MIROC"}:
        # "months since 1979-1-15"
        n = ds.sizes[time_dim]
        dates = pd.date_range("1979-01-01", periods=n, freq="MS")
        ds = ds.assign_coords({time_dim: dates})
    # LMDZ is handled separately, after selecting the region, for better performance
    return ds


def _select_jfma(da: xr.DataArray, time_dim: str) -> xr.DataArray:
    """Select JFMA months within [START_YEAR, END_YEAR]."""
    t = da[time_dim]
    mask = (
        (t.dt.year >= START_YEAR)
        & (t.dt.year <= END_YEAR)
        & t.dt.month.isin(list(JFMA))
    )
    return da.isel({time_dim: mask.values})


def _nan_da() -> xr.DataArray:
    """Placeholder all-NaN DataArray for unavailable cloud data."""
    n_jfma = (END_YEAR - START_YEAR + 1) * len(JFMA)
    return xr.DataArray(
        np.full((n_jfma, 1, 1), np.nan),
        dims=["time", "lat", "lon"],
        attrs={"units": "%", "long_name": "Low cloud fraction (unavailable)"},
    )
