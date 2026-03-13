import polars as pl
import xarray as xr

from nc.vars import DS_638_001 as v


def construct_df(ds: xr.Dataset) -> pl.DataFrame:
    """Convert relevant variables from xarray.Dataset to a Polars DataFrame."""
    if v.time in ds:
        t = ds[v.time]
        ds = ds.assign(hours_utc=t.dt.hour + t.dt.minute / 60.0 + t.dt.second / 3600.0)

    if v.altitude in ds:
        ds = ds.assign(alt_km=ds[v.altitude] / 1000.0)

    vars_to_keep = {v.latitude, v.longitude, "hours_utc", "alt_km"}
    available_vars = list(vars_to_keep.intersection(ds.variables))

    # convert to df (reset_index flattens coordinates)
    df_pandas = ds[available_vars].to_dataframe().reset_index()
    return pl.from_pandas(df_pandas)
