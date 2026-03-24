"""
Analyze and plot particle size distributions for low-level legs.
"""

import os
from typing import Literal, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from nc.loader import PROJECT_ROOT
from microphysics.load import build_low_level_dataset, PHASE_ICE
from microphysics.size_distribution import (
    SizeDistribution,
    aggregate_size_distribution,
    compute_moment,
    compute_distribution_statistics,
    bin_by_water_path,
)

PLOTS_DIR = os.path.join(
    PROJECT_ROOT, "output/microphysics_beta/plots/size_distributions"
)


def plot_mean_size_distribution(
    sd: SizeDistribution,
    output_path: str,
    log_scale: Tuple[bool, bool] = (True, True),
    include_uncertainty: bool = False,
) -> str:
    """Plot mean dN/dD vs D with optional uncertainty bands."""
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(
        sd.bin_centers_um,
        sd.dNdD,
        linewidth=2,
        color="tab:blue",
        label=r"Mean $\partial N/\partial D$",
    )

    if include_uncertainty and "std_dNdD" in sd.metadata:
        std = sd.metadata["std_dNdD"]
        ax.fill_between(
            sd.bin_centers_um,
            sd.dNdD - std,
            sd.dNdD + std,
            alpha=0.2,
            color="tab:blue",
            label=r"$\pm$ 1 std",
        )

    if log_scale[0]:
        ax.set_xscale("log")
    if log_scale[1]:
        ax.set_yscale("log")

    ax.set_xlabel(r"Diameter ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"$\partial N/\partial D$ (#/m$^4$)", fontsize=12)
    ax.set_title("Mean Particle Size Distribution", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    n_samples = sd.metadata.get("n_samples", "?")
    ax.text(
        0.02,
        0.98,
        f"n = {n_samples} samples",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_size_distribution_scatter(
    df: pl.DataFrame,
    variable: Literal["WVP", "LWP"] = "WVP",
    output_path: str = None,
    log_scale: Tuple[bool, bool] = (True, True),
) -> str:
    """Scatter plot of individual size distributions colored by WVP/LWP."""
    all_D = []
    all_dNdD = []
    all_var = []

    for row in df.iter_rows(named=True):
        conc = np.array(row["concentration"])
        bin_centers = np.array(row["bin_centers"])
        var_value = row[variable]

        if var_value is None or np.isnan(var_value):
            continue

        all_D.extend(bin_centers)
        all_dNdD.extend(conc)
        all_var.extend([var_value] * len(bin_centers))

    all_D = np.array(all_D)
    all_dNdD = np.array(all_dNdD)
    all_var = np.array(all_var)

    # filter out non-positive values for log scale
    if log_scale[1]:
        mask = all_dNdD > 0
        all_D = all_D[mask]
        all_dNdD = all_dNdD[mask]
        all_var = all_var[mask]

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(
        all_D,
        all_dNdD,
        c=all_var,
        cmap="viridis",
        s=1,
        alpha=0.3,
    )

    cbar = fig.colorbar(scatter, ax=ax)
    var_label = "WVP (g/m$^2$)" if variable == "WVP" else "LWP (g/m$^2$)"
    cbar.set_label(var_label, fontsize=12)

    if log_scale[0]:
        ax.set_xscale("log")
    if log_scale[1]:
        ax.set_yscale("log")

    ax.set_xlabel(r"Diameter ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"$\partial N/\partial D$ (#/m$^4$)", fontsize=12)
    ax.set_title(f"Size Distributions Colored by {variable}", fontsize=14)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_binned_size_distributions(
    binned_data: List[Tuple[str, pl.DataFrame]],
    variable: Literal["WVP", "LWP"] = "WVP",
    output_path: str = None,
    log_scale: Tuple[bool, bool] = (True, True),
) -> str:
    """Plot size distributions stratified by WVP/LWP bins."""
    fig, ax = plt.subplots(figsize=(10, 7))

    cmap = plt.cm.plasma
    colors = [cmap(i / len(binned_data)) for i in range(len(binned_data))]

    for (bin_label, df_subset), color in zip(binned_data, colors):
        all_conc = []
        for row in df_subset.iter_rows(named=True):
            all_conc.append(np.array(row["concentration"]))

        if not all_conc:
            continue

        first_row = df_subset.row(0, named=True)
        bin_centers = np.array(first_row["bin_centers"])
        bin_widths = np.array(first_row["bin_widths"])

        conc_array = np.column_stack(all_conc)
        sd = aggregate_size_distribution(
            conc_array, bin_centers, bin_widths, method="mean"
        )

        # normalize by dividing dN/dD by total N
        total_N = np.nansum(sd.dNdD * sd.bin_widths_um)
        dNdD_norm = sd.dNdD / total_N if total_N > 0 else sd.dNdD

        ax.plot(
            sd.bin_centers_um,
            dNdD_norm,
            linewidth=2,
            color=color,
            label=f"{bin_label} g/m$^2$",
        )

    if log_scale[0]:
        ax.set_xscale("log")
    if log_scale[1]:
        ax.set_yscale("log")

    ax.set_xlabel(r"Diameter ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"Normalized $\partial N/\partial D$ ($\mu$m$^{-1}$)", fontsize=12)
    ax.set_title(f"Size Distributions Stratified by {variable}", fontsize=14)
    ax.legend(title=f"{variable} bins", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_size_distribution_heatmap(
    segments: list,
    bin_centers: np.ndarray,
    output_path: str,
    vmin: float,
    vmax: float,
    title: str,
) -> str:
    """Time-resolved heatmap: D vs time colored by ln(dN/dD)."""
    n_segments = len(segments)
    width_ratios = [
        mdates.date2num(seg[2][-1]) - mdates.date2num(seg[2][0]) for seg in segments
    ]

    fig, axes = plt.subplots(
        1,
        n_segments,
        sharey=True,
        figsize=(max(14, 3 * n_segments), 6),
        width_ratios=width_ratios,
        squeeze=False,
    )
    axes = axes[0]

    mesh = None
    for ax, (label, concentration, times) in zip(axes, segments):
        times_num = mdates.date2num(times)
        # add small offset to avoid log(0) = -inf
        conc_log = np.log(concentration + 1e-10)

        mesh = ax.pcolormesh(
            times_num,
            bin_centers,
            conc_log,
            cmap="viridis",
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.set_xlabel("Time (UTC)", fontsize=12)
        ax.set_title(f"Leg {label}", fontsize=12)

    axes[0].set_ylabel(r"Diameter ($\mu$m)", fontsize=12)

    fig.suptitle(title, fontsize=14)
    cbar = fig.colorbar(mesh, ax=axes.tolist())
    cbar.set_label(r"$\ln(\partial N/\partial D)$ [#/m$^4$]", fontsize=12)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_integrated_properties(
    df: pl.DataFrame, output_path: str, variable: Literal["WVP", "LWP"] = "WVP"
) -> str:
    """Multi-panel scatter plots of integrated properties vs WVP or LWP."""
    M0_values = []
    M3_values = []
    Deff_values = []
    var_values = []
    flights = []

    for row in df.iter_rows(named=True):
        conc = np.array(row["concentration"])[:, np.newaxis]
        bin_centers = np.array(row["bin_centers"])
        bin_widths = np.array(row["bin_widths"])
        var_value = row[variable]
        flight = row["flight"]

        if var_value is None or np.isnan(var_value):
            continue

        M0 = compute_moment(conc, bin_centers, bin_widths, moment=0)[0]
        M2 = compute_moment(conc, bin_centers, bin_widths, moment=2)[0]
        M3 = compute_moment(conc, bin_centers, bin_widths, moment=3)[0]
        D_eff = M3 / M2 if M2 > 0 else np.nan

        M0_values.append(M0)
        M3_values.append(M3)
        Deff_values.append(D_eff)
        var_values.append(var_value)
        flights.append(flight)

    df_plot = pd.DataFrame(
        {
            variable: var_values,
            "M0": M0_values,
            "M3": M3_values,
            "D_eff": Deff_values,
            "flight": flights,
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    var_label = r"WVP (g/m$^2$)" if variable == "WVP" else r"LWP (g/m$^2$)"

    for ax_i, (ycol, ylabel, title_prefix) in zip(
        axes,
        [
            ("M0", r"$M_0$ (#/m$^3$)", "Total Number Concentration"),
            ("D_eff", r"$D_{\mathrm{eff}}$ ($\mu$m)", "Effective Diameter"),
            ("M3", r"$M_3$ ($\mu$m$^3$/m$^3$)", "Volume-Weighted Moment"),
        ],
    ):
        sns.scatterplot(
            data=df_plot,
            x=variable,
            y=ycol,
            hue="flight",
            alpha=0.6,
            s=20,
            ax=ax_i,
        )
        ax_i.set_xlabel(var_label, fontsize=11)
        ax_i.set_ylabel(ylabel, fontsize=11)
        ax_i.set_title(f"{title_prefix} vs {variable}", fontsize=12)
        ax_i.set_xscale("log")
        ax_i.set_yscale("log")
        ax_i.grid(True, alpha=0.3)
        ax_i.legend(title="Flight", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = build_low_level_dataset(phase_filter=frozenset({PHASE_ICE}))

    # get bin info from first row
    first_row = df.row(0, named=True)
    bin_centers = np.array(first_row["bin_centers"])
    bin_widths = np.array(first_row["bin_widths"])

    # concatenate all concentration arrays
    all_conc = []
    for row in df.iter_rows(named=True):
        all_conc.append(np.array(row["concentration"]))

    concentration = np.column_stack(all_conc)

    sd_mean = aggregate_size_distribution(
        concentration, bin_centers, bin_widths, method="mean"
    )
    stats = compute_distribution_statistics(concentration, bin_centers, bin_widths)

    wvp_binned = bin_by_water_path(df, variable="WVP", n_bins=5, method="quantile")
    lwp_binned = bin_by_water_path(df, variable="LWP", n_bins=5, method="quantile")

    print("Generating plots...")
    plot_mean_size_distribution(sd_mean, os.path.join(PLOTS_DIR, "mean_dNdD_vs_D.png"))

    for metric in ["WVP", "LWP"]:
        plot_size_distribution_scatter(
            df,
            variable=metric,
            output_path=os.path.join(PLOTS_DIR, f"dNdD_vs_D_colored_by_{metric}.png"),
        )
        plot_integrated_properties(
            df,
            output_path=os.path.join(
                PLOTS_DIR, f"integrated_properties_vs_{metric}.png"
            ),
            variable=metric,
        )

    plot_binned_size_distributions(
        wvp_binned,
        variable="WVP",
        output_path=os.path.join(PLOTS_DIR, "dNdD_vs_D_stratified_by_WVP.png"),
    )
    plot_binned_size_distributions(
        lwp_binned,
        variable="LWP",
        output_path=os.path.join(PLOTS_DIR, "dNdD_vs_D_stratified_by_LWP.png"),
    )

    # compute global concentration range for consistent color scale
    # add small offset to avoid log(0) = -inf
    conc_min = np.log(np.nanmin(concentration + 1e-10))
    conc_max = np.log(np.nanmax(concentration + 1e-10))

    # generate heatmap for each flight, with one panel per low-level leg
    flights = df["flight"].unique().sort()
    for flight in flights:
        df_flight = df.filter(df["flight"] == flight)
        if df_flight.is_empty():
            continue

        segments = []
        for seg_id in df_flight["segment_id"].unique().sort():
            df_seg = df_flight.filter(df_flight["segment_id"] == seg_id)
            times = []
            conc_list = []
            for row in df_seg.iter_rows(named=True):
                times.append(row["time"])
                conc_list.append(np.array(row["concentration"]))
            segments.append((seg_id, np.column_stack(conc_list), np.array(times)))

        # sort segments by start time
        segments.sort(key=lambda s: s[2][0])

        plot_size_distribution_heatmap(
            segments,
            bin_centers,
            os.path.join(PLOTS_DIR, f"dNdD_heatmap_{flight}.png"),
            vmin=conc_min,
            vmax=conc_max,
            title=f"{flight}: Size distribution on low-level legs",
        )


if __name__ == "__main__":
    main()
