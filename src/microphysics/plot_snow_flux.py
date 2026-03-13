"""
Analyze and plot snow mass flux relationships.
"""

import os
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl
import seaborn as sns

from nc.loader import PROJECT_ROOT
from microphysics.snow_mass_flux import build_flux_dataset

PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/microphysics_beta/plots/snow_flux")


def plot_flux_vs_mcao(df: pl.DataFrame, output_path: str) -> str:
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

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_binned_flux(df: pl.DataFrame, output_path: str) -> str:
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
        return output_path

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

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_normalized_flux_vs_mcao(df: pl.DataFrame, output_path: str) -> str:
    """Scatterplot of MCAO vs S/LWP and S/WVP."""
    df_valid = df.filter(
        pl.col("S_over_LWP").is_not_null() & pl.col("S_over_WVP").is_not_null()
    )

    df_pd = df_valid.to_pandas()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    sns.scatterplot(
        data=df_pd,
        x="MCAO",
        y="S_over_LWP",
        hue="flight",
        alpha=0.5,
        s=20,
        ax=ax1,
    )
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$S$/LWP (hr$^{-1}$)", fontsize=12)
    ax1.set_title(r"$S$/LWP vs MCAO (low-level legs)", fontsize=13)
    ax1.legend(title="Flight", fontsize=8)
    ax1.grid(True, alpha=0.3)

    sns.scatterplot(
        data=df_pd,
        x="MCAO",
        y="S_over_WVP",
        hue="flight",
        alpha=0.5,
        s=20,
        ax=ax2,
    )
    ax2.set_yscale("log")
    ax2.set_xlabel("MCAO $(K)$", fontsize=12)
    ax2.set_ylabel(r"$S$/WVP (hr$^{-1}$)", fontsize=12)
    ax2.set_title(r"$S$/WVP vs MCAO (low-level legs)", fontsize=13)
    ax2.legend(title="Flight", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_snow_rate_normalized_timeseries(
    df: pl.DataFrame,
    flight: str,
    output_path: str,
    ylim: Tuple[float, float] = None,
) -> str:
    """
    Time series of S/LWP and S/WVP for each low-level leg of a flight.
    """
    df_flight = df.filter(pl.col("flight") == flight)
    if df_flight.is_empty():
        return output_path

    # partition once into per-segment dicts; sort time within each segment
    partitions = df_flight.sort("time").partition_by("segment_id", as_dict=True)
    seg_data = {}
    seg_durations_ns = []
    for (seg_id,), df_seg in sorted(partitions.items()):
        times = df_seg["time"].to_numpy()
        seg_data[seg_id] = (
            times,
            df_seg["S_over_LWP"].to_numpy(),
            df_seg["S_over_WVP"].to_numpy(),
        )
        if len(times) >= 2:
            seg_durations_ns.append(int(times[-1]) - int(times[0]))

    seg_ids = list(seg_data.keys())
    max_duration_ns = max(seg_durations_ns) if seg_durations_ns else 0

    n_segs = len(seg_ids)
    fig, axes = plt.subplots(
        1, n_segs, sharey=True, figsize=(max(10, 4 * n_segs), 5), squeeze=False
    )
    axes = axes[0]

    for ax, seg_id in zip(axes, seg_ids):
        times, s_lwp, s_wvp = seg_data[seg_id]

        ax.plot(times, s_lwp, color="tab:blue", linewidth=1.2, label=r"$S$/LWP")
        ax.plot(times, s_wvp, color="tab:orange", linewidth=1.2, label=r"$S$/WVP")

        # enforce consistent time span across subplots within this flight
        if len(times) >= 1:
            t0 = times[0]
            t1 = t0 + np.timedelta64(max_duration_ns, "ns")
            ax.set_xlim(t0, t1)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.set_xlabel("Time (UTC)", fontsize=11)
        ax.set_title(f"Leg {seg_id}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_yscale("log")
    if ylim is not None:
        axes[0].set_ylim(ylim)

    axes[0].set_ylabel(r"Snow rate / water path (hr$^{-1}$)", fontsize=10)
    fig.suptitle(f"{flight}: Snow rate normalized by LWP and WVP", fontsize=13)
    plt.tight_layout()

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = build_flux_dataset()

    print("Generating plots...")

    out_flux = os.path.join(PLOTS_DIR, "snow_flux_vs_mcao.png")
    plot_flux_vs_mcao(df, out_flux)
    print(f"Saved: {out_flux}")

    out_binned = os.path.join(PLOTS_DIR, "binned_snow_flux_vs_mcao.png")
    plot_binned_flux(df, out_binned)
    print(f"Saved: {out_binned}")

    out_norm = os.path.join(PLOTS_DIR, "normalized_flux_vs_mcao.png")
    plot_normalized_flux_vs_mcao(df, out_norm)
    print(f"Saved: {out_norm}")

    # compute global y-limits for S/LWP and S/WVP across all flights
    s_over_lwp = df["S_over_LWP"].drop_nulls().to_numpy()
    s_over_wvp = df["S_over_WVP"].drop_nulls().to_numpy()
    combined = np.concatenate([s_over_lwp, s_over_wvp])
    combined = combined[combined > 0]
    ylim = (float(combined.min()), float(combined.max()))

    flights = df["flight"].unique().sort().to_list()
    for flight in flights:
        out_ts = os.path.join(
            PLOTS_DIR, f"snow_rate_normalized_timeseries_{flight}.png"
        )
        plot_snow_rate_normalized_timeseries(df, flight, out_ts, ylim=ylim)
        print(f"Saved: {out_ts}")

    print("\nDone.")


if __name__ == "__main__":
    main()
