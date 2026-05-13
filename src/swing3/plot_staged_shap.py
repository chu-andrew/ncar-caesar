"""SHAP visualizations for staged PE analysis: beeswarms, R^2, heatmap, MCAO, OOS scatter."""

import io
import os

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import shap

from nc.loader import PROJECT_ROOT
from swing3.config import MODELS, STAGED_MODELS
from swing3.features import load_grid_coords
from swing3.shap_analysis import run_staged_analysis, run_staged_oos_predictions
from swing3.types import OOSResult, StagedResult

SHAP_PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/swing3/plots/shap")


def plot_beeswarms(all_results: dict[str, dict[str, StagedResult]]) -> None:
    for model_name, stages in all_results.items():
        stage_names = list(stages.keys())
        n = len(stage_names)
        images = []
        for i, stage_name in enumerate(stage_names):
            result = stages[stage_name]
            r2 = result["r2_test_mean"]
            label = stage_name.split(": ", 1)[1] if ": " in stage_name else stage_name
            show_colorbar = i == n - 1

            with plt.rc_context(
                {"font.size": 18, "xtick.labelsize": 14, "ytick.labelsize": 14}
            ):
                shap.plots.beeswarm(
                    result["shap_values"],
                    show=False,
                    max_display=15,
                    color_bar=show_colorbar,
                    plot_size=(7, 10),
                )
                shap_fig = plt.gcf()
                shap_fig.suptitle(f"{label}\n$R^2={r2:.3f}$", fontsize=20, y=1.0)

            buf = io.BytesIO()
            shap_fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(shap_fig)
            buf.seek(0)
            images.append(plt.imread(buf))

        fig, axes = plt.subplots(1, n, figsize=(7 * n, 10))
        for ax, img in zip(axes, images):
            ax.imshow(img)
            ax.axis("off")

        fig.suptitle(model_name, fontsize=24)
        plt.subplots_adjust(top=0.95, wspace=0)

        out = os.path.join(SHAP_PLOTS_DIR, f"{model_name}_beeswarms.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


def plot_r2_by_stage(all_results: dict[str, dict[str, StagedResult]]) -> None:
    """Line plot of test R^2 at each stage per model, showing marginal gain per predictor group."""
    os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)

    model_names = list(all_results.keys())
    stage_names = [name for name, _ in STAGED_MODELS]
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(model_names)))
    x_all = np.arange(len(stage_names))

    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name, color in zip(model_names, colors):
        stages = all_results[model_name]
        xs, means, stds = [], [], []
        for xi, stage_name in enumerate(stage_names):
            if stage_name not in stages:
                continue
            xs.append(xi)
            means.append(stages[stage_name]["r2_test_mean"])
            stds.append(stages[stage_name]["r2_test_std"])
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.plot(xs, means_arr, color=color, marker="o", linewidth=1.8, label=model_name)
        ax.fill_between(
            xs, means_arr - stds_arr, means_arr + stds_arr, color=color, alpha=0.15
        )

    ax.set_xticks(x_all)
    ax.set_xticklabels(
        [s.split(": ", 1)[1] if ": " in s else s for s in stage_names], fontsize=10
    )
    ax.set_ylabel("Mean test R^2", fontsize=12)
    ax.set_title("Test R^2 by stage (marginal gain per predictor group)", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(SHAP_PLOTS_DIR, "r2_by_stage.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_intermodel_heatmap(all_results: dict[str, dict[str, StagedResult]]) -> None:
    model_names = list(all_results.keys())
    final_stages = {m: list(all_results[m].keys())[-1] for m in model_names}

    seen: set[str] = set()
    feature_names: list[str] = []
    for m in model_names:
        for f in all_results[m][final_stages[m]]["feature_names"]:
            if f not in seen:
                feature_names.append(f)
                seen.add(f)
    feat_idx = {f: j for j, f in enumerate(feature_names)}

    matrix = np.zeros((len(model_names), len(feature_names)))
    for i, model_name in enumerate(model_names):
        stage = final_stages[model_name]
        sv = all_results[model_name][stage]["shap_values"]
        mean_abs = np.abs(sv.values).mean(axis=0)
        row_sum = mean_abs.sum()
        normed = mean_abs / row_sum if row_sum > 0 else mean_abs
        for val, feat in zip(normed, all_results[model_name][stage]["feature_names"]):
            matrix[i, feat_idx[feat]] = val

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=11)

    for i in range(len(model_names)):
        for j in range(len(feature_names)):
            val = matrix[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if val > 0.15 else "black",
            )

    fig.colorbar(im, ax=ax, label="Normalized mean |SHAP|", shrink=0.8)
    ax.set_title("Relative feature importance across models (Stage 4)", fontsize=14)

    out = os.path.join(SHAP_PLOTS_DIR, "intermodel_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_mcao_dependence(all_results: dict[str, dict[str, StagedResult]]) -> None:
    model_names = list(all_results.keys())
    n_cols = 4
    norm = mcolors.Normalize(vmin=0, vmax=100)
    cmap = shap.plots.colors.red_blue

    for stage_idx, (stage_name, _) in enumerate(STAGED_MODELS, start=1):
        plot_data = []
        all_mcao_vals, all_shap_vals = [], []
        for model_name in model_names:
            if stage_name not in all_results[model_name]:
                continue
            result = all_results[model_name][stage_name]
            sv = result["shap_values"]
            X_full = result["X_full"]
            feature_names = result["feature_names"]

            mcao_idx = feature_names.index("mcao")
            mcao_x = X_full["mcao"].values
            mcao_shap = sv.values[:, mcao_idx]

            cloud = None
            if "low_cloud" in feature_names:
                c = X_full["low_cloud"].values
                if np.any(np.isfinite(c)):
                    cloud = c

            all_mcao_vals.append(mcao_x)
            all_shap_vals.append(mcao_shap)
            plot_data.append((model_name, mcao_x, mcao_shap, cloud))

        if not plot_data:
            continue

        x_lo, x_hi = np.percentile(np.concatenate(all_mcao_vals), [0, 100])
        y_lo, y_hi = np.percentile(np.concatenate(all_shap_vals), [0, 100])

        n_panels = len(plot_data)
        n_rows_stage = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows_stage,
            n_cols,
            figsize=(4.5 * n_cols, 4 * n_rows_stage),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        axes_flat = axes.flatten()
        sc_mappable = None

        for idx, (model_name, mcao_x, mcao_shap, cloud) in enumerate(plot_data):
            ax = axes_flat[idx]
            _, col = divmod(idx, n_cols)

            if cloud is not None:
                sc = ax.scatter(
                    mcao_x,
                    mcao_shap,
                    c=cloud,
                    cmap=cmap,
                    norm=norm,
                    s=4,
                    alpha=0.6,
                    linewidths=0,
                    rasterized=True,
                )
                if sc_mappable is None:
                    sc_mappable = sc
            else:
                ax.scatter(
                    mcao_x,
                    mcao_shap,
                    color="gray",
                    s=4,
                    alpha=0.4,
                    linewidths=0,
                    rasterized=True,
                )

            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.grid(True, alpha=0.3)
            ax.set_title(model_name, fontsize=16)
            ax.set_title(f"(n={len(mcao_x):,})", fontsize=10, loc="right", color="gray")

            if idx >= (n_rows_stage - 1) * n_cols:
                ax.set_xlabel("MCAO (K)", fontsize=12)
            if col == 0:
                ax.set_ylabel("SHAP value for MCAO", fontsize=12)

        for idx in range(n_panels, n_rows_stage * n_cols):
            axes_flat[idx].set_visible(False)

        if sc_mappable is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.70])
            cb = fig.colorbar(sc_mappable, cax=cbar_ax, orientation="vertical")
            cb.set_label("Low cloud fraction (%)", fontsize=12)
            cb.ax.tick_params(labelsize=10)

        fig.suptitle(
            f"SHAP dependence in stage {stage_idx}: MCAO (colored by low cloud fraction)",
            fontsize=16,
            y=0.98,
        )
        plt.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.2)

        out = os.path.join(SHAP_PLOTS_DIR, f"mcao_dependence_s{stage_idx}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


def plot_pe_scatter_by_stage(
    staged_results: dict[str, dict[str, StagedResult]],
    oos_results: dict[str, dict[str, OOSResult]],
) -> None:
    """Scatter of climate model PE vs XGBoost OOS prediction, rows=stages, cols=models.

    An extra bar-chart row below each column summarises cumulative R^2 at each stage
    so readers can gauge model skill at a glance.
    """
    os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)

    model_names = list(staged_results.keys())
    stage_names = [name for name, _ in STAGED_MODELS]
    stage_labels = [s.split(": ", 1)[1] if ": " in s else s for s in stage_names]
    stage_colors = ["#d73027", "#4575b4", "#1a9850", "#762a83"]
    n_scatter_rows, n_cols = len(stage_names), len(model_names)

    all_vals = np.concatenate(
        [
            arr
            for m in model_names
            for s in oos_results[m]
            for arr in (oos_results[m][s]["y_true"], oos_results[m][s]["y_oos_pred"])
        ]
    )
    lo, hi = float(np.percentile(all_vals, 1)), float(np.percentile(all_vals, 99))

    all_r2 = [
        staged_results[m][s]["r2_test_mean"]
        for m in model_names
        for s in stage_names
        if s in staged_results[m]
    ]
    r2_ymax = max(all_r2) * 1.12

    fig = plt.figure(figsize=(3 * n_cols, 3 * n_scatter_rows + 2.2))
    gs = gridspec.GridSpec(
        n_scatter_rows + 1,
        n_cols,
        height_ratios=[3] * n_scatter_rows + [2],
        hspace=0.08,
        wspace=0.05,
    )

    scatter_axes = np.array(
        [
            [fig.add_subplot(gs[r, c]) for c in range(n_cols)]
            for r in range(n_scatter_rows)
        ]
    )
    bar_axes = [fig.add_subplot(gs[n_scatter_rows, c]) for c in range(n_cols)]

    # Share x/y across scatter panels
    ref = scatter_axes[0, 0]
    for ax in scatter_axes.ravel():
        if ax is not ref:
            ax.sharex(ref)
            ax.sharey(ref)

    for row, (stage_name, stage_label) in enumerate(zip(stage_names, stage_labels)):
        for col, model_name in enumerate(model_names):
            ax = scatter_axes[row, col]

            if stage_name not in oos_results[model_name]:
                ax.set_visible(False)
                continue

            y_true = oos_results[model_name][stage_name]["y_true"]
            y_pred = oos_results[model_name][stage_name]["y_oos_pred"]
            r2 = staged_results[model_name][stage_name]["r2_test_mean"]

            ax.scatter(
                y_true,
                y_pred,
                s=1,
                alpha=0.4,
                color="steelblue",
                linewidths=0,
                rasterized=True,
            )
            ax.plot([lo, hi], [lo, hi], color="k", linewidth=0.8, linestyle="--")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.text(
                0.97,
                0.04,
                f"$R^2={r2:.2f}$",
                transform=ax.transAxes,
                fontsize=7,
                ha="right",
                va="bottom",
            )

            if row == 0:
                ax.set_title(model_name, fontsize=11, pad=4)
            if col == 0:
                ax.set_ylabel(stage_label, fontsize=10)

            # suppress tick labels on inner panels
            if row < n_scatter_rows - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            if col > 0:
                plt.setp(ax.get_yticklabels(), visible=False)

    # Bar summary row
    for col, model_name in enumerate(model_names):
        ax = bar_axes[col]
        heights, colors = [], []
        for stage_name, color in zip(stage_names, stage_colors):
            if stage_name not in staged_results[model_name]:
                continue
            heights.append(staged_results[model_name][stage_name]["r2_test_mean"])
            colors.append(color)
        ax.bar(
            range(len(heights)),
            heights,
            color=colors,
            width=0.7,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_ylim(0, r2_ymax)
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(labelsize=7)
        if col > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            ax.set_ylabel("$R^2$", fontsize=9)

    fig.supxlabel("Simulated PE (%)", fontsize=13, y=0.01)
    fig.supylabel("Reconstructed PE, held-out (%)", fontsize=13, x=0.01)
    fig.suptitle(
        "Precipitation efficiency: simulated vs. reconstructed", fontsize=16, y=1.005
    )

    out = os.path.join(SHAP_PLOTS_DIR, "pe_scatter_by_stage.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_stage_r2_bars(all_results: dict[str, dict[str, StagedResult]]) -> None:
    """Bar chart of test R^2 at each stage, one panel per model, shared y-axis."""
    os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)

    model_names = list(all_results.keys())
    stage_names = [name for name, _ in STAGED_MODELS]
    stage_labels = [s.split(": ", 1)[1] if ": " in s else s for s in stage_names]
    stage_colors = ["#d73027", "#4575b4", "#1a9850", "#762a83"]

    all_r2 = [
        all_results[m][s]["r2_test_mean"]
        for m in model_names
        for s in stage_names
        if s in all_results[m]
    ]
    y_max = max(all_r2) * 1.08

    n_cols = len(model_names)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 3.5), sharey=True)

    for ax, model_name in zip(axes, model_names):
        stages = all_results[model_name]
        xs, heights, colors = [], [], []
        for xi, (stage_name, color) in enumerate(zip(stage_names, stage_colors)):
            if stage_name not in stages:
                continue
            xs.append(xi)
            heights.append(stages[stage_name]["r2_test_mean"])
            colors.append(color)
        ax.bar(
            range(len(xs)),
            heights,
            color=colors,
            width=0.7,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(
            [stage_labels[i] for i in xs], rotation=40, ha="right", fontsize=7
        )
        ax.set_title(model_name, fontsize=10)
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3, axis="y")
        if ax is axes[0]:
            ax.set_ylabel("Test $R^2$", fontsize=10)

    fig.suptitle("Stage-by-stage $R^2$ (Thermo / +Dyn / +Cloud / +Iso)", fontsize=11)
    fig.tight_layout()

    out = os.path.join(SHAP_PLOTS_DIR, "stage_r2_bars.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_spatial_residuals(all_results: dict[str, dict[str, StagedResult]]) -> None:
    """Time-mean spatial map of Stage 4 residuals (actual - predicted PE) per model."""
    from swing3.plot_mcao_pe_map import setup_map

    os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)

    model_names = list(all_results.keys())
    final_stage = {m: list(all_results[m].keys())[-1] for m in model_names}

    residual_maps: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for model_name in model_names:
        lat, lon, mask = load_grid_coords(model_name)
        n_lat, n_lon = len(lat), len(lon)
        n_flat = len(mask)
        n_time = n_flat // (n_lat * n_lon)

        residuals = all_results[model_name][final_stage[model_name]]["residuals"]
        full = np.full(n_flat, np.nan)
        full[mask] = residuals
        spatial_mean = np.nanmean(full.reshape(n_time, n_lat, n_lon), axis=0)
        residual_maps[model_name] = (lat, lon, spatial_mean)

    all_res = np.concatenate([v for _, _, v in residual_maps.values()])
    clim = float(np.nanpercentile(np.abs(all_res), 98))

    n_cols = 4
    n_rows = (len(model_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 2.8 * n_rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )

    im_last = None
    for idx, model_name in enumerate(model_names):
        ax = axes[idx // n_cols, idx % n_cols]
        lat, lon, res_map = residual_maps[model_name]
        setup_map(ax)
        im_last = ax.pcolormesh(
            lon,
            lat,
            res_map,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=-clim,
            vmax=clim,
        )
        ax.set_title(model_name, fontsize=11)

    for idx in range(len(model_names), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.subplots_adjust(right=0.88, hspace=0.15, wspace=0.1)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    fig.colorbar(im_last, cax=cbar_ax).set_label(
        "Actual - Predicted PE (%)", fontsize=11
    )
    fig.suptitle("Stage 4 time-mean residuals (JFMA, CAESAR domain)", fontsize=13)

    out = os.path.join(SHAP_PLOTS_DIR, "spatial_residuals.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)

    all_results = {m: run_staged_analysis(m) for m in MODELS}

    plot_beeswarms(all_results)
    plot_r2_by_stage(all_results)
    plot_stage_r2_bars(all_results)
    plot_intermodel_heatmap(all_results)
    plot_mcao_dependence(all_results)
    plot_spatial_residuals(all_results)

    oos_results = {m: run_staged_oos_predictions(m) for m in MODELS}
    plot_pe_scatter_by_stage(all_results, oos_results)


if __name__ == "__main__":
    main()
