import os
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl
import seaborn as sns

from nc.flights import LOW_LEVEL_LEGS
from nc.loader import PROJECT_ROOT
from microphysics.snow_mass_flux import build_flux_dataset, filter_legs

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
    """Scatterplot of MCAO vs S/LWP, S/WVP, and S/VMR_VXL."""
    df_pd = df.to_pandas()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

    panels = [
        (
            ax1,
            "S_over_LWP",
            r"$S$/LWP (hr$^{-1}$)",
            r"$S$/LWP vs MCAO (low-level legs)",
        ),
        (
            ax2,
            "S_over_WVP",
            r"$S$/WVP (hr$^{-1}$)",
            r"$S$/WVP vs MCAO (low-level legs)",
        ),
        (
            ax3,
            "S_over_VMR_VXL",
            r"$S$ / VMR_VXL (kg m$^{-2}$ s$^{-1}$ ppmv$^{-1}$)",
            r"$S$ / VMR_VXL vs MCAO (low-level legs)",
        ),
    ]

    for ax, ycol, ylabel, title in panels:
        df_plot = df_pd[df_pd[ycol].notna()]
        sns.scatterplot(
            data=df_plot,
            x="MCAO",
            y=ycol,
            hue="flight",
            alpha=0.5,
            s=20,
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(title="Flight", fontsize=8)
        ax.grid(True, alpha=0.3)

    ax3.set_xlabel("MCAO $(K)$", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_normalized_flux_by_altitude(
    df: pl.DataFrame, output_path: str, title_suffix: str = "All Flights"
) -> str:
    """Scatterplot of S/WVP, S/LWP, S/VMR_VXL vs MCAO colored by flight altitude."""
    df_valid = df.filter(
        pl.col("MCAO").is_not_null()
        & ~pl.col("MCAO").is_nan()
        & pl.col("alt_insitu").is_not_null()
        & ~pl.col("alt_insitu").is_nan()
    ).to_pandas()

    panels = [
        ("S_over_WVP", r"$S$/WVP (hr$^{-1}$)", r"$S$/WVP vs MCAO"),
        ("S_over_LWP", r"$S$/LWP (hr$^{-1}$)", r"$S$/LWP vs MCAO"),
        (
            "S_over_VMR_VXL",
            r"$S$ / VMR_VXL (kg m$^{-2}$ s$^{-1}$ ppmv$^{-1}$)",
            r"$S$ / VMR_VXL vs MCAO",
        ),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

    alt_min = df_valid["alt_insitu"].min()
    alt_max = df_valid["alt_insitu"].max()

    for ax, (ycol, ylabel, title) in zip(axes, panels):
        df_plot = df_valid[df_valid[ycol].notna() & (df_valid[ycol] > 0)]
        sc = ax.scatter(
            df_plot["MCAO"],
            df_plot[ycol],
            c=df_plot["alt_insitu"],
            cmap="viridis",
            vmin=alt_min,
            vmax=alt_max,
            alpha=0.5,
            s=15,
        )
        ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{title} ({title_suffix})", fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, label="Altitude (m)")

    axes[-1].set_xlabel("MCAO $(K)$", fontsize=12)
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

    out_alt = os.path.join(PLOTS_DIR, "normalized_flux_vs_mcao_by_altitude.png")
    plot_normalized_flux_by_altitude(df, out_alt)
    print(f"Saved: {out_alt}")

    # per-leg plots
    for flight, legs in LOW_LEVEL_LEGS.items():
        for start, end in legs:
            label = f"{flight.lower()}_{start}-{end}"
            df_leg = filter_legs(df, {flight: [(start, end)]})
            if df_leg.is_empty():
                continue

            out = os.path.join(
                PLOTS_DIR, f"normalized_flux_vs_mcao_by_altitude_{label}.png"
            )
            plot_normalized_flux_by_altitude(
                df_leg, out, title_suffix=f"{flight} Leg {start}-{end}"
            )
            print(f"Saved: {out}")

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
