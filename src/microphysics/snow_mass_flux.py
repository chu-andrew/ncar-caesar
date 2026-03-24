"""
Compute snow mass flux using Szyrmer & Zawadzki (2010) parameterization
and analyze relationship with MCAO index.
"""

import numpy as np
import polars as pl

from ds_638_021.mcao import build_merged_dataset
from microphysics.load import build_low_level_dataset, PHASE_ICE
from nc.loader import open_dataset
from nc.units import um_to_m, S_PER_HR, KG_PER_G
from nc.vars import DS_638_001 as v001


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
    D_m = um_to_m(bin_centers)
    dD_m = um_to_m(bin_widths)

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


def load_insitu_ancillary(flight: str) -> pl.DataFrame:
    """Load VMR_VXL, altitude, latitude, and longitude from 638-001 for a flight."""
    with open_dataset(v001.dataset, flight) as ds:
        times = ds[v001.time].values
        vmr = ds[v001.vmr_vxl].values
        alt = ds[v001.altitude].values
        lat = ds[v001.latitude].values
        lon = ds[v001.longitude].values

    return pl.DataFrame(
        {
            "time": times,
            "VMR_VXL": vmr.astype(np.float64),
            "alt_insitu": alt.astype(np.float64),
            "lat": lat.astype(np.float64),
            "lon": lon.astype(np.float64),
        }
    )


def filter_legs(
    df: pl.DataFrame, legs: dict[str, list[tuple[int, int]]]
) -> pl.DataFrame:
    """
    Filter flux dataset to specific flight legs.

    Args:
        df: output of build_flux_dataset()
        legs: e.g. {"RF07": [(8, 9), (14, 15)]}
    """
    keep = [
        (flight, f"{s}-{e}") for flight, leg_list in legs.items() for s, e in leg_list
    ]
    if not keep:
        return df.filter(pl.lit(False))
    flights, segment_ids = zip(*keep)
    return df.filter(
        pl.struct("flight", "segment_id").is_in(
            [{"flight": f, "segment_id": sid} for f, sid in zip(flights, segment_ids)]
        )
    )


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

    # join VMR_VXL and altitude from 638-001 per flight
    print("Loading VMR_VXL and altitude from in-situ data...")
    flights = df["flight"].unique().sort().to_list()
    frames = []
    for flight in flights:
        df_flight = df.filter(pl.col("flight") == flight)
        df_vmr = load_insitu_ancillary(flight)
        df_flight = df_flight.sort("time").join_asof(
            df_vmr.sort("time"),
            on="time",
            strategy="nearest",
            tolerance="2s",
        )
        frames.append(df_flight)
    df = pl.concat(frames)

    # normalized snow rates (hr^-1)
    # S is kg/m^2/s, LWP and WVP are g/m^2; convert to kg/m^2/1000
    # then S/WP = 1/s; multiply by 3600 to get 1/hr
    # ==> S / WP * 1000 * 3600
    s_wp_to_per_hr = S_PER_HR / KG_PER_G
    df = df.with_columns(
        pl.when(pl.col("LWP") > 0)
        .then(pl.col("S") / pl.col("LWP") * s_wp_to_per_hr)
        .alias("S_over_LWP"),
        pl.when(pl.col("WVP") > 0)
        .then(pl.col("S") / pl.col("WVP") * s_wp_to_per_hr)
        .alias("S_over_WVP"),
        pl.when(pl.col("VMR_VXL") > 0)
        .then(pl.col("S") / pl.col("VMR_VXL"))
        .alias("S_over_VMR_VXL"),
    )

    n_vmr = df.filter(
        pl.col("VMR_VXL").is_not_null() & ~pl.col("VMR_VXL").is_nan()
    ).height
    print(
        f"\nMerged dataset: {df.height} timesteps with valid S and MCAO"
        f"\n\tS range: [{df['S'].min():.3e}, {df['S'].max():.3e}] kg/m^2/s"
        f"\n\tMCAO range: [{df['MCAO'].min():.2f}, {df['MCAO'].max():.2f}] K"
        f"\n\tVMR_VXL: {n_vmr}/{df.height} rows matched"
    )

    return df
