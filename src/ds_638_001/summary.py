import polars as pl
import xarray as xr


LATITUDE = "LATC"
LONGITUDE = "LONC"
ZAXIS = "GGALT"
TIME = "Time"


def construct_df(ds: xr.Dataset) -> pl.DataFrame:
    """Convert relevant variables from xarray.Dataset to a Polars DataFrame."""
    if TIME in ds:
        t = ds[TIME]
        ds = ds.assign(hours_utc=t.dt.hour + t.dt.minute / 60.0 + t.dt.second / 3600.0)

    if ZAXIS in ds:
        ds = ds.assign(alt_km=ds[ZAXIS] / 1000.0)

    vars_to_keep = {LATITUDE, LONGITUDE, "hours_utc", "alt_km"}
    available_vars = list(vars_to_keep.intersection(ds.variables))

    # convert to df (reset_index flattens coordinates)
    df_pandas = ds[available_vars].to_dataframe().reset_index()
    return pl.from_pandas(df_pandas)
