import os
import tempfile
from typing import NamedTuple

from cdo import Cdo
import numpy as np
import pandas as pd
import xarray as xr

from nc.cache import MEMORY
from nc.loader import open_file
from nc.remote import CLOUD_DIR, SWING3_MODELS
from nc.vars import SWING3 as v
from nc.vars import SWING3_LMDZ as v_lmdz
from swing3.models import _crop_region, JFMA, START_YEAR, END_YEAR

CDO = Cdo()

MODELS = list(SWING3_MODELS.keys())



class CloudSpec(NamedTuple):
    glob: str
    cloud_cover_var: str
    scale: float  # multiply to get percent (0-100)
    pressure_gt_hpa: (
        int | None
    )  # select levels with pressure > this value (hPa), then take column maximum


CLOUD_VAR_MAP: dict[str, CloudSpec] = {
    "CAM6": CloudSpec("*CLDLOW*", "CLDLOW", 100.0, None),
    "ECHAM": CloudSpec("*low_cld*", "low_cld", 100.0, None),
    "GISS": CloudSpec("*CloudCover*", "pcldl", 1.0, None),
    "GSM": CloudSpec("*CloudCover*", "tcdclcl", 1.0, None),
    "LMDZ": CloudSpec("*CloudCover*", "cldl", 100.0, None),
    "MIROC": CloudSpec("*CloudCover*", "cldfrc", 1.0, 680),
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

        if spec.pressure_gt_hpa is not None:
            da = da.sel({v.pressure: da[v.pressure] > spec.pressure_gt_hpa}).max(
                dim=v.pressure
            )
        if model == "LMDZ":
            # aggregate daily data to monthly using arithmetic mean
            da = da.resample({time_dim: "MS"}).mean()
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
    """JFMA low-cloud fraction time series regridded to the T42 grid using conservative remapping."""
    da = load_low_cloud(model)
    lat_t42, lon_t42 = _t42_grid()

    if model not in CLOUD_VAR_MAP:
        return da

    # Skip regridding if the data is already on the T42 grid
    if (
        da.lat.size == lat_t42.size
        and da.lon.size == lon_t42.size
        and np.allclose(da.lat.values, lat_t42)
        and np.allclose(da.lon.values, lon_t42)
    ):
        return da

    da = _prep_for_cdo(da).rename("cld")
    with tempfile.TemporaryDirectory() as tmpdir:
        in_file = os.path.join(tmpdir, "input.nc")
        grid_file = os.path.join(tmpdir, "grid.txt")
        da.to_netcdf(in_file)
        with open(grid_file, "w") as f:
            f.write(_cdo_grid_desc(lat_t42, lon_t42))
        return CDO.remapcon(grid_file, input=in_file, returnXArray="cld")


def load_low_cloud_clim_t42(model: str) -> xr.DataArray:
    """JFMA climatological mean low-cloud fraction regridded to the T42 grid."""
    da = load_low_cloud_t42(model)
    time_dim = v_lmdz.time if v_lmdz.time in da.dims else v.time
    return da.mean(dim=time_dim)


@MEMORY.cache
def _t42_grid() -> tuple[np.ndarray, np.ndarray]:
    """T42 grid lat/lon from a reference SWING3 model, cropped to CAESAR_BOUNDS."""
    ref_model = MODELS[0]
    time_dim = v_lmdz.time if ref_model == "LMDZ" else v.time
    with open_file(SWING3_MODELS[ref_model], decode_times=False) as ds:
        ref = _crop_region(ds[v.precip_efficiency].isel({time_dim: 0}))
        return ref.lat.values, ref.lon.values


def _prep_for_cdo(da: xr.DataArray) -> xr.DataArray:
    """Add CF attributes required by CDO to recognise the spatial grid."""
    da = da.copy()
    da["lat"].attrs.update({"units": "degrees_north", "axis": "Y"})
    da["lon"].attrs.update({"units": "degrees_east", "axis": "X"})
    if v_lmdz.time in da.dims:
        da = da.rename({v_lmdz.time: "time"})
    return da


def _cdo_grid_desc(lat: np.ndarray, lon: np.ndarray) -> str:
    """CDO grid description string for a regular lat/lon grid."""
    return (
        f"gridtype = lonlat\n"
        f"xsize    = {lon.size}\n"
        f"ysize    = {lat.size}\n"
        f"xfirst   = {lon[0]:.6f}\n"
        f"xinc     = {np.diff(lon).mean():.6f}\n"
        f"yfirst   = {lat[0]:.6f}\n"
        f"yinc     = {np.diff(lat).mean():.6f}\n"
    )


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
