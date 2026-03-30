import os
from typing import Tuple

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl
import seaborn as sns
from scipy.stats import gaussian_kde

from ds_638_001.plot_summary import setup_map
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


def plot_pe_map(
    df: pl.DataFrame,
    pe_col: str,
    pe_label: str,
    output_path: str,
    title_suffix: str = "All Flights",
) -> str:
    df_pd = df.filter(
        pl.col(pe_col).is_not_null()
        & (pl.col(pe_col) > 0)
        & pl.col("alt_insitu").is_not_null()
        & pl.col("lat").is_not_null()
        & pl.col("lon").is_not_null()
    ).to_pandas()

    log_pe = np.log(df_pd[pe_col])
    lon = df_pd["lon"].to_numpy()
    lat = df_pd["lat"].to_numpy()

    map_proj = ccrs.LambertConformal(
        central_longitude=float(df_pd["lon"].mean()),
        central_latitude=float(df_pd["lat"].mean()),
    )
    transform = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection=map_proj)
    ax2 = fig.add_subplot(1, 2, 2, projection=map_proj)

    for ax in (ax1, ax2):
        setup_map(ax)

    sc1 = ax1.scatter(
        lon, lat, c=log_pe, cmap="plasma", alpha=0.7, s=12, transform=transform
    )
    fig.colorbar(sc1, ax=ax1, label=f"ln({pe_label})", shrink=0.7)
    ax1.set_title(pe_label)

    sc2 = ax2.scatter(
        lon,
        lat,
        c=df_pd["alt_insitu"],
        cmap="viridis",
        alpha=0.7,
        s=12,
        transform=transform,
    )
    fig.colorbar(sc2, ax=ax2, label="Altitude (m)", shrink=0.7)
    ax2.set_title("Altitude")

    fig.suptitle(f"{pe_label} and Altitude: {title_suffix}", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _hexbin_log_vs_mcao_by_altitude(
    df: pl.DataFrame,
    configs: list,
    suptitle: str,
    output_path: str,
    n_alt_bins: int = 4,
) -> str:
    """Generic hexbin of log(col) vs MCAO faceted by altitude bin.

    configs: list of (col_name, ylabel, label) tuples.
    """
    df_base = df.filter(
        pl.col("MCAO").is_not_null()
        & ~pl.col("MCAO").is_nan()
        & pl.col("alt_insitu").is_not_null()
        & ~pl.col("alt_insitu").is_nan()
    )
    df_base, bin_edges, bin_labels = _altitude_bins(df_base, n_alt_bins)
    actual_n_bins = len(bin_edges) - 1

    mcao_lim = (float(df_base["MCAO"].min()), float(df_base["MCAO"].max()))

    # cache per-row log-transformed dataframes and y-limits
    df_cache = {}
    log_lims = {}
    for row, (col_name, _, _) in enumerate(configs):
        df_row = df_base.filter(
            pl.col(col_name).is_not_null() & (pl.col(col_name) > 0)
        ).with_columns(pl.col(col_name).log().alias("log_val"))
        df_cache[row] = df_row
        log_vals = df_row["log_val"].to_numpy()
        log_lims[row] = (np.nanmin(log_vals), np.nanmax(log_vals))

    n_rows = len(configs)
    n_cols = actual_n_bins
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), squeeze=False
    )

    all_hb_objects = {}  # (row, col) -> (hb, ax, n)

    for row, (col_name, ylabel, _) in enumerate(configs):
        df_row = df_cache[row]
        log_lim = log_lims[row]

        for col in range(actual_n_bins):
            ax = axes[row, col]
            df_bin = df_row.filter(pl.col("alt_bin") == col)
            n = len(df_bin)

            hb = ax.hexbin(
                df_bin["MCAO"].to_numpy(),
                df_bin["log_val"].to_numpy(),
                gridsize=30,
                cmap="inferno",
                mincnt=1,
                extent=[*mcao_lim, *log_lim],
            )
            counts = hb.get_array()
            hb.set_array(counts / counts.sum())  # normalize per plot
            all_hb_objects[(row, col)] = (hb, ax, n)
            ax.grid(True, alpha=0.2, color="white")
            ax.set_xlim(mcao_lim)
            ax.set_ylim(log_lim)

            if row == 0:
                ax.set_title(bin_labels[col], fontsize=15)
            ax.set_title(f"n={n}", fontsize=9, loc="right", color="gray")

            if row == n_rows - 1:
                ax.set_xlabel("MCAO (K)", fontsize=13)

            if col == 0:
                ax.set_ylabel(ylabel, fontsize=13)

    if all_hb_objects:
        global_vmax = max(hb.get_array().max() for hb, _, _ in all_hb_objects.values())
        for hb, _, _ in all_hb_objects.values():
            hb.set_clim(0, global_vmax)

        last_hb = all_hb_objects[max(all_hb_objects)][0]
        fig.subplots_adjust(right=0.88, top=0.93)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.75])
        fig.colorbar(last_hb, cax=cbar_ax, label="Fraction of observations")

    fig.suptitle(suptitle, fontsize=18)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_pe_vs_mcao_hexbin(
    df: pl.DataFrame,
    output_path: str,
    n_alt_bins: int = 4,
) -> str:
    return _hexbin_log_vs_mcao_by_altitude(
        df,
        configs=[
            ("S_over_LWP", r"ln(S/LWP) (hr$^{-1}$)", r"$S$/LWP"),
            ("S_over_WVP", r"ln(S/WVP) (hr$^{-1}$)", r"$S$/WVP"),
            (
                "S_over_VMR_VXL",
                r"ln(S/VMR_VXL) (kg m$^{-2}$ s$^{-1}$ ppmv$^{-1}$)",
                r"$S$/VMR",
            ),
        ],
        suptitle=r"$\ln$(PE) vs MCAO by altitude bin",
        output_path=output_path,
        n_alt_bins=n_alt_bins,
    )


def plot_raw_vs_mcao_hexbin(
    df: pl.DataFrame,
    output_path: str,
    n_alt_bins: int = 4,
) -> str:
    """2D hexbin of log(raw variable) vs MCAO by altitude bin for S, LWP, WVP, VMR_VXL."""
    return _hexbin_log_vs_mcao_by_altitude(
        df,
        configs=[
            ("S", r"ln(S) (kg m$^{-2}$ s$^{-1}$)", "S"),
            ("LWP", r"ln(LWP) (g m$^{-2}$)", "LWP"),
            ("WVP", r"ln(WVP) (g m$^{-2}$)", "WVP"),
            ("VMR_VXL", r"ln(VMR_VXL) (ppmv)", "VMR_VXL"),
        ],
        suptitle=r"$\ln$(raw variables) vs MCAO by altitude bin",
        output_path=output_path,
        n_alt_bins=n_alt_bins,
    )


def _altitude_bins(df: pl.DataFrame, n_bins: int):
    quantile_pts = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(np.quantile(df["alt_insitu"].to_numpy(), quantile_pts))
    actual_n = len(bin_edges) - 1
    labels = [f"{bin_edges[i]:.0f}–{bin_edges[i + 1]:.0f} m" for i in range(actual_n)]
    alt_bins = np.clip(
        np.digitize(df["alt_insitu"].to_numpy(), bin_edges) - 1, 0, actual_n - 1
    )
    return df.with_columns(pl.Series("alt_bin", alt_bins)), bin_edges, labels


def plot_kde_by_altitude_bin(
    df: pl.DataFrame,
    output_path: str,
    n_alt_bins: int = 4,
) -> str:
    """
    KDE distributions of MCAO and log(PE) stratified by altitude bins.

    Each panel shows one KDE curve per altitude bin plus an overall background,
    with mean +- std markers.
    """
    df_base = df.filter(
        pl.col("MCAO").is_not_null()
        & ~pl.col("MCAO").is_nan()
        & pl.col("alt_insitu").is_not_null()
        & ~pl.col("alt_insitu").is_nan()
    )
    df_base, _, bin_labels = _altitude_bins(df_base, n_alt_bins)
    actual_n_bins = len(bin_labels)

    pe_cols = ["S_over_LWP", "S_over_WVP", "S_over_VMR_VXL"]
    pe_labels = [r"$S$/LWP", r"$S$/WVP", r"$S$/VMR"]

    colors = plt.cm.plasma(np.linspace(0.1, 0.85, actual_n_bins))
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    def _draw_kde_panel(ax, all_vals, bin_val_list, xlabel, title):
        all_vals = all_vals[np.isfinite(all_vals)]
        bin_val_list = [v[np.isfinite(v)] for v in bin_val_list]
        x_range = np.linspace(all_vals.min(), all_vals.max(), 300)
        kde_all = gaussian_kde(all_vals)
        y_all = kde_all(x_range)
        ax.fill_between(x_range, y_all, alpha=0.15, color="gray")
        ax.plot(x_range, y_all, color="gray", alpha=0.5, linewidth=1.5, label="All")

        y_peak = y_all.max()
        marker_ys = np.linspace(y_peak * 1.20, y_peak * 1.80, actual_n_bins)

        for i, (vals, color) in enumerate(zip(bin_val_list, colors)):
            kde_bin = gaussian_kde(vals)
            ax.plot(
                x_range, kde_bin(x_range), color=color, linewidth=2, label=bin_labels[i]
            )
            mean, std = np.mean(vals), np.std(vals)
            ax.errorbar(
                mean,
                marker_ys[i],
                xerr=std,
                fmt="o",
                color=color,
                markersize=8,
                capsize=5,
                linewidth=2,
                zorder=5,
            )

        ax.set_ylim(bottom=0, top=y_peak * 2.00)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # MCAO
    mcao_all = df_base["MCAO"].to_numpy()
    mcao_by_bin = [
        df_base.filter(pl.col("alt_bin") == i)["MCAO"].to_numpy()
        for i in range(actual_n_bins)
    ]
    _draw_kde_panel(axes[0], mcao_all, mcao_by_bin, "MCAO (K)", "MCAO")

    # log(PE) variables
    for ax, pe_col, pe_label in zip(axes[1:], pe_cols, pe_labels):
        df_pe = df_base.filter(
            pl.col(pe_col).is_not_null() & (pl.col(pe_col) > 0)
        ).with_columns(pl.col(pe_col).log().alias("log_pe"))

        all_vals = df_pe["log_pe"].to_numpy()
        by_bin = [
            df_pe.filter(pl.col("alt_bin") == i)["log_pe"].to_numpy()
            for i in range(actual_n_bins)
        ]
        _draw_kde_panel(ax, all_vals, by_bin, rf"$\ln$({pe_label})", pe_label)

    fig.suptitle("MCAO and PE distributions by altitude bin", fontsize=13)
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

    out_hexbin = os.path.join(PLOTS_DIR, "pe_vs_mcao_hexbin_by_altitude.png")
    plot_pe_vs_mcao_hexbin(df, out_hexbin)
    print(f"Saved: {out_hexbin}")

    out_raw_hexbin = os.path.join(PLOTS_DIR, "raw_vs_mcao_hexbin_by_altitude.png")
    plot_raw_vs_mcao_hexbin(df, out_raw_hexbin)
    print(f"Saved: {out_raw_hexbin}")

    out_kde = os.path.join(PLOTS_DIR, "kde_by_altitude.png")
    plot_kde_by_altitude_bin(df, out_kde)
    print(f"Saved: {out_kde}")

    pe_panels = [
        ("S_over_LWP", r"$S$/LWP"),
        ("S_over_WVP", r"$S$/WVP"),
        ("S_over_VMR_VXL", r"$S$/VMR"),
    ]

    for pe_col, pe_label in pe_panels:
        out = os.path.join(PLOTS_DIR, f"pe_map_{pe_col.lower()}.png")
        plot_pe_map(df, pe_col, pe_label, out)
        print(f"Saved: {out}")

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
