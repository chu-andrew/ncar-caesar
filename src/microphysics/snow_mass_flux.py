"""
Compute snow mass flux using Szyrmer & Zawadzki (2010) parameterization
and analyze relationship with MCAO index.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from nc.loader import PROJECT_ROOT
from ds_638_021.mcao import build_merged_dataset
from microphysics.data_loader import build_low_level_dataset, PHASE_ICE

PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/microphysics_beta/plots/snow_flux")


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

    print(
        f"\nMerged dataset: {df.height} timesteps with valid S and MCAO"
        f"\n\tS range: [{df['S'].min():.3e}, {df['S'].max():.3e}] kg/m^2/s"
        f"\n\tMCAO range: [{df['MCAO'].min():.2f}, {df['MCAO'].max():.2f}] K"
    )

    return df


def plot_flux_vs_mcao(df: pl.DataFrame) -> None:
    """Scatter plot of snow mass flux vs MCAO."""
    df_pd = df.to_pandas()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=df_pd,
        x="MCAO",
        y="S",
        hue="flight",
        alpha=0.6,
        s=20,
        ax=ax,
    )

    ax.set_xlabel("MCAO $(K)$", fontsize=12)
    ax.set_ylabel(r"Snow Mass Flux $S$ (kg/m$^2$/s)", fontsize=12)
    ax.set_title("Snow Mass Flux vs MCAO Index", fontsize=14)
    ax.set_yscale("log")
    ax.legend(title="Flight", fontsize=9)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(PLOTS_DIR, "snow_flux_vs_mcao.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_binned_flux(df: pl.DataFrame) -> None:
    """Binned mean snow flux vs MCAO with error bars."""
    df_valid = df.filter(pl.col("MCAO").is_not_null() & ~pl.col("MCAO").is_nan())

    mcao_vals = df_valid["MCAO"].to_numpy()
    flux_vals = df_valid["S"].to_numpy()

    bin_edges = np.arange(
        np.floor(mcao_vals.min()),
        np.ceil(mcao_vals.max()) + 1,
        1.0,
    )

    bin_idx = np.digitize(mcao_vals, bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    means = []
    stds = []
    valid_centers = []

    for i in range(len(bin_centers)):
        mask = bin_idx == i
        if mask.sum() >= 2:
            means.append(np.nanmean(flux_vals[mask]))
            stds.append(np.nanstd(flux_vals[mask]))
            valid_centers.append(bin_centers[i])

    if not valid_centers:
        print("No valid bins for flux plot.")
        return

    centers = np.array(valid_centers)
    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(
        centers,
        means,
        yerr=stds,
        fmt="o-",
        color="tab:blue",
        linewidth=2,
        markersize=8,
        capsize=5,
    )

    ax.set_xlabel("MCAO $(K)$", fontsize=12)
    ax.set_ylabel(r"Snow Mass Flux $S$ (kg/m$^2$/s)", fontsize=12)
    ax.set_title(r"Binned Snow Flux vs MCAO (mean $\pm$ std)", fontsize=14)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(PLOTS_DIR, "binned_snow_flux_vs_mcao.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = build_flux_dataset()

    print("Generating plots...")
    plot_flux_vs_mcao(df)
    plot_binned_flux(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
