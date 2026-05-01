import metpy.calc as mpcalc
import numpy as np
import pandas as pd
from metpy.units import units as munits

from nc.cache import MEMORY
from nc.loader import open_file
from nc.remote import SWING3_MODELS
from nc.vars import SWING3 as v
from nc.vars import SWING3_LMDZ as v_lmdz
from nc.vars import SWING3_SST as v_sst
from swing3.clouds import END_YEAR, START_YEAR, load_low_cloud_t42
from swing3.models import P_850, crop_region, jfma_indices
from swing3.omega import load_omega
from swing3.sst import load_sst

P_700 = 700  # hPa
P_925 = 925  # hPa


@MEMORY.cache
def _load_model_arrays(model: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Load all raw feature arrays and the PE target for one model.

    Returns (arrays, time_groups) where every value in arrays is a raveled 1-D
    numpy array. time_groups encodes the time-step index of each sample for use
    with GroupShuffleSplit. omega_925 and low_cloud may contain NaN.
    """
    sst_da = load_sst()
    time_dim = v_lmdz.time if model == "LMDZ" else v.time
    sst_region = crop_region(sst_da)
    n_sst = sst_da.sizes[v_sst.time]
    n_cloud = (END_YEAR - START_YEAR + 1) * 12

    with open_file(SWING3_MODELS[model], decode_times=False) as ds:
        n_times = ds.sizes[time_dim]
        n_min = min(n_times, n_sst, n_cloud)
        jfma = jfma_indices(n_min)

        def _get(var_name, p_level=None):
            da = ds[var_name]
            if p_level is not None:
                da = da.sel({v.pressure: p_level}, method="nearest")
            return crop_region(da.isel({time_dim: jfma})).load().values

        # temperature: load full 3D once, slice both levels in memory
        t_3d = crop_region(ds[v.temperature].isel({time_dim: jfma})).load()
        t_850 = t_3d.sel({v.pressure: P_850}).values
        t_700 = t_3d.sel({v.pressure: P_700}).values
        del t_3d

        theta850 = (
            mpcalc.potential_temperature(P_850 * munits.hPa, t_850 * munits.degC)
            .to("K")
            .magnitude
        )
        mcao = sst_region.values[jfma] - theta850

        # dD vapor: load full 3D once, slice both layers in memory
        dD_3d = crop_region(ds[v.dD_vapor].isel({time_dim: jfma})).load()
        dD_ft = dD_3d.sel({v.pressure: slice(800, 600)}).mean(dim=v.pressure).values
        dD_bl = dD_3d.sel({v.pressure: slice(925, 800)}).mean(dim=v.pressure).values
        del dD_3d

        raw = {
            "mcao": mcao,
            "sh": _get(v.surface_specific_humidity),
            "qvsum": _get(v.precipitable_water),
            "q_700": _get(v.specific_humidity, P_700),
            "t_700": t_700,
            "ts": _get(v.surface_temperature),
            "wind_sfc": mpcalc.wind_speed(
                # use 925 hPa instead of 1000 hPa, since 925 hPa is consistently defined across all models
                _get(v.u_wind, P_925) * munits("m/s"),
                _get(v.v_wind, P_925) * munits("m/s"),
            )
            .to("m/s")
            .magnitude,
            "iuq": _get(v.moisture_flux_u),
            "ivq": _get(v.moisture_flux_v),
            "dD_gradient": dD_ft - dD_bl,
            "dDp": _get(v.dD_precip),
            "dexcessp": _get(v.dexcess_precip),
            "dDs": _get(v.dD_surface),
            "dexcesss": _get(v.dexcess_surface),
            "pref": _get(v.precip_efficiency),
        }

    # time groups for block GroupShuffleSplit (prevent spatial autocorrelation leakage)
    time_groups = (
        np.broadcast_to(np.arange(mcao.shape[0])[:, None, None], mcao.shape)
        .ravel()
        .copy()
    )

    # TODO: verify time ranges for omega
    omega = load_omega(model, n_times=n_min)
    raw["omega_925"] = omega.sel(p=P_925).values
    raw["omega_700"] = omega.sel(p=P_700).values

    # TODO: verify time ranges for low cloud
    low_cloud = load_low_cloud_t42(model)
    assert low_cloud.size >= mcao.size, (
        f"[{model}] low_cloud ({low_cloud.size} elements) is smaller than "
        f"mcao ({mcao.size} elements); time ranges are mismatched."
    )
    raw["low_cloud"] = low_cloud.values.ravel()[: mcao.size]

    return {k: arr.ravel() for k, arr in raw.items()}, time_groups


@MEMORY.cache
def load_shap_features(model: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Assemble predictor DataFrame and PE target for model, with NaN rows removed."""
    arrays, time_groups = _load_model_arrays(model)

    target = arrays["pref"]
    feature_arrays = {k: arr for k, arr in arrays.items() if k != "pref"}

    valid_pe = np.isfinite(target)
    retained_pe = valid_pe & (target >= 0) & (target <= 100)
    frac_removed = 100 * (1 - retained_pe.sum() / max(1, valid_pe.sum()))
    print(
        f"[{model}] PE outside [0, 100] filter removed {frac_removed:.2f}% of valid PE points."
    )

    nan_ok = {
        "low_cloud",
        "omega_925",
    }  # XGBoost handles NaN; partial coverage expected
    mask = retained_pe
    for col, arr in feature_arrays.items():
        if col not in nan_ok:
            n_lost = (retained_pe & ~np.isfinite(arr)).sum()
            if n_lost > 0:
                print(
                    f"[{model}] ({col}): {n_lost:,} NaN samples ({100 * n_lost / max(1, retained_pe.sum()):.2f}%)"
                )
            mask &= np.isfinite(arr)

    n_total = valid_pe.sum()
    n_kept = mask.sum()
    print(
        f"[{model}] {n_kept:,} / {n_total:,} samples retained after all filters "
        f"({100 * n_kept / max(1, n_total):.1f}%)"
    )

    features = pd.DataFrame({k: arr[mask] for k, arr in feature_arrays.items()})
    return features.reset_index(drop=True), target[mask], time_groups[mask]
