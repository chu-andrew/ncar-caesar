from dataclasses import dataclass
from typing import Literal, List, Tuple

import numpy as np
import polars as pl


@dataclass
class SizeDistribution:
    """Container for aggregated size distribution data."""

    bin_centers_um: np.ndarray
    bin_widths_um: np.ndarray
    dNdD: np.ndarray  # #/m^4
    N_total_per_bin: np.ndarray  # #/m^3
    metadata: dict


def aggregate_size_distribution(
    concentration: np.ndarray,
    bin_centers: np.ndarray,
    bin_widths: np.ndarray,
    method: Literal["mean", "median", "sum"] = "mean",
) -> SizeDistribution:
    """Aggregate size distribution across time."""
    if method == "mean":
        dNdD = np.nanmean(concentration, axis=1)
    elif method == "median":
        dNdD = np.nanmedian(concentration, axis=1)
    elif method == "sum":
        dNdD = np.nansum(concentration, axis=1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    # bin_widths is in um; convert to m to match dNdD (#/m^4) -> N (#/m^3)
    N_total_per_bin = dNdD * (bin_widths * 1e-6)

    metadata = {
        "method": method,
        "n_samples": concentration.shape[1],
    }

    return SizeDistribution(
        bin_centers_um=bin_centers,
        bin_widths_um=bin_widths,
        dNdD=dNdD,
        N_total_per_bin=N_total_per_bin,
        metadata=metadata,
    )


def compute_distribution_statistics(
    concentration: np.ndarray, bin_centers: np.ndarray, bin_widths: np.ndarray
) -> dict:
    """Compute ensemble statistics on size distributions."""
    mean_dNdD = np.nanmean(concentration, axis=1)
    median_dNdD = np.nanmedian(concentration, axis=1)
    std_dNdD = np.nanstd(concentration, axis=1)
    p25_dNdD = np.nanpercentile(concentration, 25, axis=1)
    p75_dNdD = np.nanpercentile(concentration, 75, axis=1)

    M_0 = compute_moment(concentration, bin_centers, bin_widths, moment=0)
    M_2 = compute_moment(concentration, bin_centers, bin_widths, moment=2)
    M_3 = compute_moment(concentration, bin_centers, bin_widths, moment=3)

    with np.errstate(divide="ignore", invalid="ignore"):
        D_eff = M_3 / M_2
        D_eff = np.where(np.isfinite(D_eff), D_eff, np.nan)

    return {
        "mean_dNdD": mean_dNdD,
        "median_dNdD": median_dNdD,
        "std_dNdD": std_dNdD,
        "p25_dNdD": p25_dNdD,
        "p75_dNdD": p75_dNdD,
        "M_0": M_0,
        "M_2": M_2,
        "M_3": M_3,
        "effective_diameter": D_eff,
    }


def bin_by_water_path(
    df: pl.DataFrame,
    variable: Literal["WVP", "LWP"] = "WVP",
    n_bins: int = 5,
    method: Literal["quantile", "uniform"] = "quantile",
) -> List[Tuple[str, pl.DataFrame]]:
    """Stratify data into bins by WVP or LWP."""
    df_valid = df.filter(pl.col(variable).is_not_null() & ~pl.col(variable).is_nan())

    if df_valid.is_empty():
        return []

    vals = df_valid[variable].to_numpy()

    if method == "quantile":
        bin_edges = np.quantile(vals, np.linspace(0, 1, n_bins + 1))
    elif method == "uniform":
        bin_edges = np.linspace(vals.min(), vals.max(), n_bins + 1)
    else:
        raise ValueError(f"Unknown binning method: {method}")

    bin_edges = np.unique(bin_edges)
    n_bins = len(bin_edges) - 1

    bin_idx = np.digitize(vals, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    df_valid = df_valid.with_columns(pl.Series("_bin_idx", bin_idx))

    binned_data = []
    for i in range(n_bins):
        df_subset = df_valid.filter(pl.col("_bin_idx") == i)
        if df_subset.is_empty():
            continue

        bin_label = f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}"
        df_subset = df_subset.drop("_bin_idx")
        binned_data.append((bin_label, df_subset))

    return binned_data


def compute_moment(
    concentration: np.ndarray,
    bin_centers: np.ndarray,
    bin_widths: np.ndarray,
    moment: int = 0,
) -> np.ndarray:
    """
    Compute k-th moment of size distribution.
    M_k = Sum D_i^k * concentration_i * Delta_D_i
    """
    D_m = bin_centers * 1e-6
    dD_m = bin_widths * 1e-6

    integrand = (D_m**moment)[:, np.newaxis] if concentration.ndim == 2 else D_m**moment
    integrand = (
        integrand * dD_m[:, np.newaxis] if concentration.ndim == 2 else integrand * dD_m
    )

    if concentration.ndim == 2:
        M = np.nansum(integrand * concentration, axis=0)
    else:
        M = np.nansum(integrand * concentration)

    # Integration was done in SI (m), so M_k has diameter units in m^k.
    # Convert diameter contribution from m^k to um^k so that e.g.
    # D_eff = M_3/M_2 comes out in um directly.
    if moment > 0:
        M = M * (1e6**moment)

    return M
