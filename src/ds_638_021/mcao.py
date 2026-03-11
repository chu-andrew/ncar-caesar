"""
Marine Cold-Air Outbreak index computation and composite plots.

Analysis is limited to low-level legs.
SST from RSTB, theta_850 from interpolated MARLi.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from nc.flights import FLIGHTS
from nc.loader import DATASET_VARS, PROJECT_ROOT, open_dataset

from ds_638_021.potential_temperature import compute_theta_850
from ds_638_038.water_path import LOW_LEVEL_LEGS
from ds_638_038.load import load_gvr_segment

PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/638-021/plots/mcao")

THETA_TOLERANCE = "30s"


def load_rstb(flight: str) -> pl.DataFrame:
    """
    Load radiometric surface temperature (RSTB) from 638-001.

    RSTB is in degC.
    """
    with open_dataset("638-001", flight) as ds:
        _vars_001 = DATASET_VARS["638-001"]
        times = ds[_vars_001["time"]].values
        rstb = ds["RSTB"].values

    return pl.DataFrame(
        {
            "time": times,
            "RSTB": rstb.astype(np.float64),
        }
    )


def load_theta850(flight: str) -> pl.DataFrame:
    """
    Load theta_850 from MARLi (638-021) with datetime64 timestamps.
    Uses regridded + interpolated values from compute_theta_850.
    Includes theta_850_std (std of boundary endpoints for interpolated gaps, NaN for measured).
    Only returns rows with valid (non-NaN) theta_850 values.
    """
    result = compute_theta_850(flight)

    # convert regridded float hours to datetime64 using the flight date
    date_str = FLIGHTS[flight]
    base = np.datetime64(date_str, "ns")
    hours = result["time_regrid_utc_hours"]
    ns_per_hour = np.int64(3_600_000_000_000)
    times = base + (hours * ns_per_hour).astype("timedelta64[ns]")

    df = pl.DataFrame(
        {
            "time": times,
            "theta_850": result["theta_850"].astype(np.float64),
            "theta_850_std": result["theta_850_std"].astype(np.float64),
        }
    )
    return df.filter(~pl.col("theta_850").is_nan())


def merge_flight_segment(
    flight: str,
    start_pt: int,
    end_pt: int,
    df_rstb: pl.DataFrame,
    df_theta: pl.DataFrame,
) -> pl.DataFrame:
    """
    Merge GVR, RSTB, and theta_850 for a low-level leg.

    theta_850 comes from the interpolated MARLi time series (which fills gaps
    during low-level legs using the mean of adjoining upper-leg boundary values).
    theta_850_std is the std of those boundary endpoints.
    """

    RSTB_TOLERANCE = "10s"

    df_gvr = load_gvr_segment(flight, start_pt, end_pt)
    if df_gvr.is_empty():
        return pl.DataFrame()

    # join RSTB onto GVR timestamps
    df = df_gvr.sort("time").join_asof(
        df_rstb.sort("time"),
        on="time",
        strategy="nearest",
        tolerance=RSTB_TOLERANCE,
    )

    # join theta_850 + theta_850_std onto GVR timestamps
    df = df.join_asof(
        df_theta.sort("time"),
        on="time",
        strategy="nearest",
        tolerance=THETA_TOLERANCE,
    )

    # compute MCAO index: SST(K) - theta_850; RSTB is in degC
    df = df.with_columns(
        (pl.col("RSTB") + 273.15).alias("SST_K"),
    ).with_columns(
        (pl.col("SST_K") - pl.col("theta_850")).alias("MCAO"),
    )

    df = df.with_columns(
        pl.lit(flight).alias("flight"),
        pl.lit(f"{start_pt}-{end_pt}").alias("segment"),
    )

    return df


def build_merged_dataset() -> pl.DataFrame:
    """Build the full merged dataset across all flights and segments."""
    frames = []

    for flight in LOW_LEVEL_LEGS:
        if flight not in FLIGHTS:
            continue
        print(f"Processing {flight}...")

        df_rstb = load_rstb(flight)
        df_theta = load_theta850(flight)

        # expand multi-segment low-level legs into unit segments
        low_level_units = set()
        for s, e in LOW_LEVEL_LEGS[flight]:
            for j in range(s, e):
                low_level_units.add((j, j + 1))

        for start_pt, end_pt in sorted(low_level_units):
            df_seg = merge_flight_segment(flight, start_pt, end_pt, df_rstb, df_theta)
            if not df_seg.is_empty():
                n_valid = df_seg.filter(
                    pl.col("MCAO").is_not_null() & ~pl.col("MCAO").is_nan()
                ).height
                mean_t = df_seg["theta_850"].mean()
                std_vals = df_seg["theta_850_std"].drop_nulls().drop_nans()
                std_t = std_vals.mean() if len(std_vals) > 0 else float("nan")
                print(
                    f"\tlow-level {start_pt}-{end_pt}: "
                    f"{n_valid}/{df_seg.height} valid MCAO, "
                    f"theta_850={mean_t:.2f} +/- {std_t:.2f} K"
                )
                frames.append(df_seg)

    if not frames:
        raise RuntimeError("no data merged: check segment definitions and data files")

    common_cols = [
        "time",
        "LWP",
        "WVP",
        "SST_K",
        "theta_850",
        "theta_850_std",
        "MCAO",
        "flight",
        "segment",
    ]
    frames = [f.select(common_cols) for f in frames]
    return pl.concat(frames)


def plot_scatter(df: pl.DataFrame) -> None:
    """Scatter plots: MCAO vs WVP, LWP, and LWP/WVP, colored by flight."""
    df_pd = df.filter(
        pl.col("MCAO").is_not_null() & ~pl.col("MCAO").is_nan()
    ).to_pandas()

    pairs = [
        ("MCAO", "WVP", "WVP $(g/m^2)$"),
        ("MCAO", "LWP", "LWP $(g/m^2)$"),
    ]

    for log_scale in [False, True]:
        for xcol, ycol, ylabel in pairs:
            if log_scale:
                df_plot = df_pd[df_pd[ycol] > 0].copy()
                df_plot[f"ln_{ycol}"] = np.log(df_plot[ycol])
                y_column = f"ln_{ycol}"
                y_label = f"ln({ycol})"
                title = f"ln({ycol}) vs MCAO (low-level legs)"
                suffix = f"ln{ycol.lower()}"
            else:
                df_plot = df_pd
                y_column = ycol
                y_label = ylabel
                title = f"{ycol} vs MCAO (low-level legs)"
                suffix = ycol.lower()

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=df_plot,
                x=xcol,
                y=y_column,
                hue="flight",
                alpha=0.5,
                s=15,
                ax=ax,
            )
            ax.set_xlabel("MCAO $(K)$")
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.legend(title="Flight", fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)

            out_path = os.path.join(PLOTS_DIR, f"scatter_mcao_vs_{suffix}.png")
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")

    # LWP/WVP ratio
    df_ratio = df_pd[df_pd["WVP"] > 0].copy()
    df_ratio["LWP_WVP"] = df_ratio["LWP"] / df_ratio["WVP"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df_ratio,
        x="MCAO",
        y="LWP_WVP",
        hue="flight",
        alpha=0.5,
        s=15,
        ax=ax,
    )
    ax.set_xlabel("MCAO $(K)$")
    ax.set_ylabel("LWP / WVP")
    ax.set_title("LWP/WVP Ratio vs MCAO (low-level legs)")
    ax.legend(title="Flight", fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(PLOTS_DIR, "scatter_mcao_vs_lwp_wvp.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_hexbin(df: pl.DataFrame) -> None:
    """2D histogram / hexbin density plots for low-level legs."""
    df_pd = df.filter(
        pl.col("MCAO").is_not_null() & ~pl.col("MCAO").is_nan()
    ).to_pandas()

    pairs = [
        ("MCAO", "WVP", "WVP $(g/m^2)$"),
        ("MCAO", "LWP", "LWP $(g/m^2)$"),
    ]

    for log_scale in [False, True]:
        for xcol, ycol, ylabel in pairs:
            x = df_pd[xcol].values
            y = df_pd[ycol].values
            mask = np.isfinite(x) & np.isfinite(y)
            if log_scale:
                mask &= y > 0
            x, y = x[mask], y[mask]
            if log_scale:
                y = np.log(y)

            if log_scale:
                y_label = f"ln({ycol})"
                title = f"ln({ycol}) vs MCAO (low-level legs)"
                suffix = f"ln{ycol.lower()}"
            else:
                y_label = ylabel
                title = f"{ycol} vs MCAO (low-level legs)"
                suffix = ycol.lower()

            fig, ax = plt.subplots(figsize=(8, 6))
            hb = ax.hexbin(x, y, gridsize=30, cmap="YlOrRd", mincnt=1)
            fig.colorbar(hb, ax=ax, label="Count")
            ax.set_xlabel("MCAO $(K)$")
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            out_path = os.path.join(PLOTS_DIR, f"hexbin_mcao_vs_{suffix}.png")
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")


def plot_binned_stats(df: pl.DataFrame) -> None:
    """Binned statistics: mean plus/minus std of WVP/LWP per MCAO bin."""
    df_valid = df.filter(pl.col("MCAO").is_not_null() & ~pl.col("MCAO").is_nan())

    mcao_vals = df_valid["MCAO"].to_numpy()
    if len(mcao_vals) == 0:
        print("No valid MCAO data for binned stats.")
        return

    bin_edges = np.arange(
        np.floor(mcao_vals.min()),
        np.ceil(mcao_vals.max()) + 1,
        1.0,
    )

    # bin and compute stats via numpy
    bin_idx = np.digitize(mcao_vals, bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for log_scale in [False, True]:
        for ycol, ylabel in [("WVP", "WVP $(g/m^2)$"), ("LWP", "LWP $(g/m^2)$")]:
            y_vals = df_valid[ycol].to_numpy()

            means = []
            stds = []
            valid_centers = []

            for i in range(len(bin_centers)):
                mask = bin_idx == i
                y_bin = y_vals[mask]
                if log_scale:
                    # filter for positive values before taking log
                    y_bin = y_bin[(y_bin > 0) & np.isfinite(y_bin)]
                    if len(y_bin) >= 2:
                        y_bin = np.log(y_bin)
                        means.append(np.mean(y_bin))
                        stds.append(np.std(y_bin))
                        valid_centers.append(bin_centers[i])
                else:
                    if mask.sum() >= 2:
                        means.append(np.nanmean(y_bin))
                        stds.append(np.nanstd(y_bin))
                        valid_centers.append(bin_centers[i])

            if not valid_centers:
                continue

            centers = np.array(valid_centers)
            means = np.array(means)
            stds = np.array(stds)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(centers, means, "o-", color="tab:blue", linewidth=2, markersize=6)
            ax.fill_between(
                centers,
                means - stds,
                means + stds,
                alpha=0.2,
                color="tab:blue",
            )
            ax.set_xlabel("MCAO $(K)$")

            if log_scale:
                ax.set_ylabel(f"ln({ycol}) $(ln(g/m^2))$")
                ax.set_title(f"ln({ycol}) vs MCAO (mean $\\pm$ std)")
                suffix = f"ln{ycol.lower()}"
            else:
                ax.set_ylabel(ylabel)
                ax.set_title(f"{ycol} vs MCAO (mean $\\pm$ std)")
                suffix = ycol.lower()

            ax.grid(True, alpha=0.3)

            out_path = os.path.join(PLOTS_DIR, f"binned_mcao_vs_{suffix}.png")
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")


def load_full_flight_altitude(flight: str) -> pl.DataFrame:
    """Load full-flight altitude from 638-001."""
    _vars_001 = DATASET_VARS["638-001"]
    with open_dataset("638-001", flight) as ds:
        times = ds[_vars_001["time"]].values
        alt = ds[_vars_001["altitude"]].values

    return pl.DataFrame(
        {
            "time": times,
            "alt": alt.astype(np.float64),
        }
    )


def plot_timeseries(df: pl.DataFrame) -> None:
    """Per-flight time series of MCAO (low-level legs) with full-flight altitude."""
    import matplotlib.dates as mdates

    df_valid = df.filter(pl.col("MCAO").is_not_null() & ~pl.col("MCAO").is_nan())
    flights = df_valid["flight"].unique().sort().to_list()

    # compute global MCAO axis limits
    mcao_min = df_valid["MCAO"].min()
    mcao_max = df_valid["MCAO"].max()

    for flight in flights:
        df_f = df_valid.filter(pl.col("flight") == flight).sort("time")
        df_pd = df_f.to_pandas()

        # full-flight altitude trace
        df_alt = load_full_flight_altitude(flight).to_pandas()

        fig, ax1 = plt.subplots(figsize=(12, 5))

        ax1.scatter(
            df_pd["time"],
            df_pd["MCAO"],
            c="tab:blue",
            s=8,
            alpha=0.6,
            label="MCAO",
        )

        ax1.set_ylabel("MCAO (K)")
        ax1.set_xlabel("Time (UTC)")
        ax1.set_ylim(mcao_min, mcao_max)

        ax2 = ax1.twinx()
        ax2.plot(
            df_alt["time"],
            df_alt["alt"],
            color="black",
            linewidth=0.8,
            alpha=0.5,
            label="Altitude",
        )
        ax2.set_ylabel("Altitude (m)")

        # combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax1.set_title(f"{flight}: MCAO and Altitude (low-level legs)")
        ax1.grid(True, alpha=0.3)

        out_path = os.path.join(PLOTS_DIR, f"timeseries_{flight.lower()}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = build_merged_dataset()
    df_valid = df.filter(pl.col("MCAO").is_not_null() & ~pl.col("MCAO").is_nan())
    n_valid = df_valid.height

    print(
        f"\nMerged dataset (low-level legs): {df.height} rows, {n_valid} with valid MCAO"
        f"\n\tMCAO range: [{df_valid['MCAO'].min():.2f}, {df_valid['MCAO'].max():.2f}] K"
    )

    plot_timeseries(df)
    plot_scatter(df)
    plot_hexbin(df)
    plot_binned_stats(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
