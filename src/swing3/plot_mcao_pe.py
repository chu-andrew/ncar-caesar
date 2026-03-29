import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from nc.loader import PROJECT_ROOT
from nc.remote import SWING3_MODELS
from swing3.models import load_mcao_pe
from swing3.sst import load_sst

PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/remote/swing3/plots")
MODELS = list(SWING3_MODELS.keys())


def plot_hexbin_by_model(
    all_data: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: str,
) -> None:
    models = list(all_data.keys())
    n_cols = 4
    n_rows = (len(models) + n_cols - 1) // n_cols

    BUFFER = 1  # to prevent hexbins from being clipped by axis edges
    mcao_lim = (
        min(d[0].min() for d in all_data.values()) - BUFFER,
        max(d[0].max() for d in all_data.values()) + BUFFER,
    )
    pe_lim = (
        min(d[1].min() for d in all_data.values()) - BUFFER,
        max(d[1].max() for d in all_data.values()) + BUFFER,
    )

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows), squeeze=False
    )

    all_hb_objects = {}  # flat index -> hb

    for idx, model in enumerate(models):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        mcao, pe = all_data[model]
        n = len(mcao)

        hb = ax.hexbin(
            mcao,
            pe,
            gridsize=30,
            cmap="inferno",
            mincnt=1,
            extent=[*mcao_lim, *pe_lim],
        )
        counts = hb.get_array()
        hb.set_array(counts / counts.sum())
        all_hb_objects[idx] = hb
        ax.grid(True, alpha=0.4, color="white")
        ax.set_xlim(mcao_lim)
        ax.set_ylim(pe_lim)

        ax.set_title(model, fontsize=16)
        ax.set_title(f"(n={n:,})", fontsize=10, loc="right", color="gray")
        ax.set_xlabel("MCAO (K)", fontsize=11)
        if col == 0:
            ax.set_ylabel("PE (%)", fontsize=11)

    for idx in range(len(models), n_rows * n_cols):
        axes[divmod(idx, n_cols)].set_visible(False)

    # shared color scale
    if all_hb_objects:
        global_vmax = max(hb.get_array().max() for hb in all_hb_objects.values())
        for hb in all_hb_objects.values():
            hb.set_clim(0, global_vmax)

        fig.subplots_adjust(right=0.90, top=0.90, hspace=0.30)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.70])
        fig.colorbar(
            all_hb_objects[len(models) - 1],
            cax=cbar_ax,
            label="Fraction of observations",
        )

    fig.suptitle("Precipitation efficiency vs MCAO by WisoMIP model", fontsize=18)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_kde_by_model(
    all_data: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: str,
) -> None:
    models = list(all_data.keys())
    n_models = len(models)
    colors = plt.cm.plasma(np.linspace(0, 0.8, n_models))

    all_mcao = np.concatenate([all_data[m][0] for m in models])
    all_pe = np.concatenate([all_data[m][1] for m in models])

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 7),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    (ax_mcao_kde, ax_pe_kde), (ax_mcao_forest, ax_pe_forest) = axes

    # share x-axis between each KDE and its forest panel
    ax_mcao_kde.sharex(ax_mcao_forest)
    ax_pe_kde.sharex(ax_pe_forest)

    def _draw_kde(ax, all_vals, per_model_vals, title):
        all_vals = all_vals[np.isfinite(all_vals)]
        x_range = np.linspace(all_vals.min(), all_vals.max(), 300)
        x_lo, x_hi = np.percentile(all_vals, [1, 99])

        kde_all = gaussian_kde(all_vals)
        y_all = kde_all(x_range)
        ax.fill_between(x_range, y_all, alpha=0.12, color="gray")
        ax.plot(x_range, y_all, color="gray", alpha=0.5, linewidth=1.5, label="All")

        for vals, color, label in zip(per_model_vals, colors, models):
            kde = gaussian_kde(vals)
            ax.plot(x_range, kde(x_range), color=color, linewidth=2, label=label)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(title, fontsize=15)
        ax.legend(fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), visible=False)

    def _draw_forest(ax, per_model_vals, xlabel):
        for i, (vals, color) in enumerate(zip(per_model_vals, colors)):
            mean, std = np.mean(vals), np.std(vals)
            ax.errorbar(
                mean,
                i,
                xerr=std,
                fmt="o",
                color=color,
                markersize=7,
                capsize=4,
                linewidth=1.8,
                zorder=5,
            )

        ax.set_yticks(range(n_models))
        ax.set_yticklabels(models, fontsize=9)
        ax.set_ylim(-0.8, n_models - 0.2)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=11)
        ax.grid(True, alpha=0.3, axis="x")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    mcao_by_model = [all_data[m][0][np.isfinite(all_data[m][0])] for m in models]
    pe_by_model = [all_data[m][1][np.isfinite(all_data[m][1])] for m in models]

    _draw_kde(ax_mcao_kde, all_mcao, mcao_by_model, "MCAO")
    _draw_kde(ax_pe_kde, all_pe, pe_by_model, "PE")
    _draw_forest(ax_mcao_forest, mcao_by_model, "MCAO (K)")
    _draw_forest(ax_pe_forest, pe_by_model, "PE (%)")

    fig.suptitle(
        "MCAO and precipitation efficiency distributions by WisoMIP model", fontsize=18
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading SST...")
    sst_da = load_sst()

    all_data = {}
    for model in MODELS:
        print(f"Loading {model}...")
        mcao, pe = load_mcao_pe(model, sst_da=sst_da)
        print(f"\t{model}: {len(mcao):,} valid points")
        all_data[model] = (mcao, pe)

    out_hexbin = os.path.join(PLOTS_DIR, "pe_vs_mcao_hexbin_by_model.png")
    plot_hexbin_by_model(all_data, out_hexbin)

    out_kde = os.path.join(PLOTS_DIR, "kde_by_model.png")
    plot_kde_by_model(all_data, out_kde)


if __name__ == "__main__":
    main()
