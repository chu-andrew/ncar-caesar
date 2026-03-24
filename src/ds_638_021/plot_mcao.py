import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from ds_638_021.mcao import build_merged_dataset
from nc.loader import PROJECT_ROOT, open_dataset
from nc.vars import DS_638_001 as v001, DS_638_021 as v

PLOTS_DIR = os.path.join(PROJECT_ROOT, f"output/{v.dataset}/plots/mcao")


def plot_scatter(df: pl.DataFrame) -> None:
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

    df_ratio_base = df_pd[df_pd["WVP"] > 0].copy()
    df_ratio_base["LWP_WVP"] = df_ratio_base["LWP"] / df_ratio_base["WVP"]
    df_ratio_ln = df_ratio_base[df_ratio_base["LWP"] > 0].copy()
    df_ratio_ln["ln_LWP_WVP"] = np.log(df_ratio_ln["LWP_WVP"])

    for df_plot, ycol, ylabel, title, suffix in [
        (
            df_ratio_base,
            "LWP_WVP",
            "LWP / WVP",
            "LWP/WVP Ratio vs MCAO (low-level legs)",
            "lwp_wvp",
        ),
        (
            df_ratio_ln,
            "ln_LWP_WVP",
            "ln(LWP / WVP)",
            "ln(LWP/WVP) vs MCAO (low-level legs)",
            "ln_lwp_wvp",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=df_plot,
            x="MCAO",
            y=ycol,
            hue="flight",
            alpha=0.5,
            s=15,
            ax=ax,
        )
        ax.set_xlabel("MCAO $(K)$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title="Flight", fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

        out_path = os.path.join(PLOTS_DIR, f"scatter_mcao_vs_{suffix}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def plot_hexbin(df: pl.DataFrame) -> None:
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
    with open_dataset("638-001", flight) as ds:
        times = ds[v001.time].values
        alt = ds[v001.altitude].values

    return pl.DataFrame(
        {
            "time": times,
            "alt": alt.astype(np.float64),
        }
    )


def plot_timeseries(df: pl.DataFrame) -> None:
    df_valid = df.filter(pl.col("MCAO").is_not_null() & ~pl.col("MCAO").is_nan())
    flights = df_valid["flight"].unique().sort().to_list()

    mcao_min = df_valid["MCAO"].min()
    mcao_max = df_valid["MCAO"].max()

    for flight in flights:
        df_f = df_valid.filter(pl.col("flight") == flight).sort("time")
        df_pd = df_f.to_pandas()

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
        f"\nMerged dataset (low-level legs): {df.height} rows, {n_valid} with valid MCAO\n"
        f"\tMCAO range: [{df_valid['MCAO'].min():.2f}, {df_valid['MCAO'].max():.2f}] K"
    )

    plot_timeseries(df)
    plot_scatter(df)
    plot_hexbin(df)
    plot_binned_stats(df)


if __name__ == "__main__":
    main()
