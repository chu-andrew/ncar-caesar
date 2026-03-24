import pandas as pd
import xarray as xr

from nc.loader import open_file
from nc.remote import ERA5_SST
from nc.vars import SWING3_SST as v


def load_sst() -> xr.DataArray:
    with open_file(ERA5_SST, decode_times=False) as ds:
        sst = ds[v.sst].load()
        times = pd.date_range("1979-01-01", periods=ds.sizes[v.time], freq="MS")
        sst = sst.assign_coords({v.time: times})
        return sst
