import polars as pl
import xarray as xr

from nc.units import m_to_km
from nc.vars import DS_638_001 as v


def construct_df(ds: xr.Dataset) -> pl.DataFrame:
    """Convert relevant variables from xarray.Dataset to a Polars DataFrame."""
    vars_to_keep = {v.latitude, v.longitude}

    if v.time in ds:
        t = ds[v.time]
        ds = ds.assign(hours_utc=t.dt.hour + t.dt.minute / 60.0 + t.dt.second / 3600.0)
        vars_to_keep.add("hours_utc")

    if v.altitude in ds:
        ds = ds.assign(alt_km=m_to_km(ds[v.altitude]))
        vars_to_keep.add("alt_km")

    df_pandas = ds[list(vars_to_keep)].to_dataframe().reset_index()
    return pl.from_pandas(df_pandas)
