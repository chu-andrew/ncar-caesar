"""
Compute snow mass flux using Szyrmer & Zawadzki (2010) parameterization
and analyze relationship with MCAO index.
"""

import numpy as np
import polars as pl

from ds_638_021.mcao import build_merged_dataset
from microphysics.load import build_low_level_dataset, PHASE_ICE


def compute_snow_mass_flux(
    concentration: np.ndarray,
    bin_centers: np.ndarray,
    bin_widths: np.ndarray,
) -> np.ndarray:
    """
    Compute snow mass flux using Szyrmer & Zawadzki (2010) formula.

    S = integral m(D) * v_t(D) * N(D) dD

    where:
        m(D) = 0.044 * D^2  (kg)
        v_t(D) = 2.29 * D^0.18  (m/s)
        N(D) = dN/dD (concentration)

    Args:
        concentration: dN/dD in #/m^4, shape (n_bins,) or (n_bins, n_times)
        bin_centers: bin centers in micrometers
        bin_widths: bin widths in micrometers

    Returns:
        S in kg/m^2/s, shape () or (n_times,)
    """
    D_m = bin_centers * 1e-6  # um -> m
    dD_m = bin_widths * 1e-6  # um -> m

    m_D = 0.044 * D_m**2  # kg (mass-size, a_m=0.044 kg/m^2)
    vt_D = 2.29 * D_m**0.18  # m/s (fall speed)

    # integrand units: kg * m/s; after multiplying by N(D) [#/m^4] and dD [m]
    # => kg/(m^2*s)
    integrand = m_D * vt_D

    if concentration.ndim == 1:
        S = np.nansum(integrand * concentration * dD_m)
    else:
        integrand_2d = integrand[:, np.newaxis]
        dD_m_2d = dD_m[:, np.newaxis]
        S = np.nansum(integrand_2d * concentration * dD_m_2d, axis=0)

    return S


def build_flux_dataset() -> pl.DataFrame:
    """Build dataset with snow mass flux and MCAO for all low-level legs."""
    print("Loading microphysics data...")
    df_micro = build_low_level_dataset(phase_filter=frozenset({PHASE_ICE}))

    print("Computing snow mass flux...")
    flux_values = []

    for row in df_micro.iter_rows(named=True):
        conc = np.array(row["concentration"])
        bin_centers = np.array(row["bin_centers"])
        bin_widths = np.array(row["bin_widths"])

        S = compute_snow_mass_flux(conc, bin_centers, bin_widths)
        flux_values.append(S)

    df_micro = df_micro.with_columns(pl.Series("S", flux_values))

    print("Loading MCAO data...")
    df_mcao = build_merged_dataset()

    print("Joining datasets...")
    df = df_micro.sort("time").join_asof(
        df_mcao.select(["time", "MCAO"]).sort("time"),
        on="time",
        strategy="nearest",
        tolerance="30s",
    )

    df = df.filter(pl.col("MCAO").is_not_null() & ~pl.col("MCAO").is_nan())

    # normalized snow rates (hr^-1)
    # S is kg/m^2/s, LWP and WVP are g/m^2; convert to kg/m^2/1000
    # then S/WP = 1/s; multiply by 3600 to get 1/hr
    # ==> S / WP * 1000 * 3600
    s_wp_to_per_hr = 3600 * 1000
    df = df.with_columns(
        pl.when(pl.col("LWP") > 0)
        .then(pl.col("S") / pl.col("LWP") * s_wp_to_per_hr)
        .alias("S_over_LWP"),
        pl.when(pl.col("WVP") > 0)
        .then(pl.col("S") / pl.col("WVP") * s_wp_to_per_hr)
        .alias("S_over_WVP"),
    )

    print(
        f"\nMerged dataset: {df.height} timesteps with valid S and MCAO"
        f"\n\tS range: [{df['S'].min():.3e}, {df['S'].max():.3e}] kg/m^2/s"
        f"\n\tMCAO range: [{df['MCAO'].min():.2f}, {df['MCAO'].max():.2f}] K"
    )

    return df
