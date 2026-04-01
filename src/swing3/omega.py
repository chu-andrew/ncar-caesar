import xarray as xr

from nc.cache import MEMORY
from nc.loader import open_file
from nc.remote import OMEGA_MODELS
from swing3.grids import crop_region
from swing3.models import jfma_indices


@MEMORY.cache
def load_omega(model: str, n_times: int | None = None) -> xr.DataArray:
    """Load JFMA omega over CAESAR_BOUNDS for the given model.

    n_times: if provided, limits how many time steps are considered before
    computing JFMA indices. Pass n_min from load_shap_features to stay
    aligned with other feature arrays.

    Coverage at pressure levels for our geographic and temporal area of interest:
        1000 hPa: 0–36%
        925 hPa: 71–100%; MIROC (71%) and CAM6 (90%) are thin
        850 hPa: 88–100%; complete except MIROC (88%)
        700 hPa: 100%
    """
    with open_file(OMEGA_MODELS[model], decode_times=False) as ds:
        n = n_times if n_times is not None else ds.sizes["time"]
        jfma = jfma_indices(n)
        omega = crop_region(ds["omega"].isel(time=jfma)).load()
    return omega
