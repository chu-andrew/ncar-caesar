"""SHAP visualizations for staged PE analysis across WisoMIP models."""

import io
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import shap

from nc.loader import PROJECT_ROOT
from swing3.config import (
    MODELS,
    STAGED_MODELS,
)
from swing3.shap_analysis import run_staged_analysis

SHAP_PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/remote/swing3/plots/shap")


def plot_beeswarms(all_results: dict[str, dict]) -> None:
    n = len(STAGED_MODELS)

    for model_name, stages in all_results.items():
        images = []
        for i, (stage_name, _) in enumerate(STAGED_MODELS):
            result = stages[stage_name]
            r2 = result["r2_test_mean"]
            label = stage_name.split(": ", 1)[1] if ": " in stage_name else stage_name
            show_colorbar = i == n - 1  # only on the last panel

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


def plot_intermodel_heatmap(all_results: dict[str, dict]) -> None:
    stage_4 = STAGED_MODELS[-1][0]
    model_names = list(all_results.keys())

    feature_names = all_results[model_names[0]][stage_4]["feature_names"]

    # Build matrix: rows=models, cols=features, values=mean|SHAP| normalized per row
    matrix = np.zeros((len(model_names), len(feature_names)))
    for i, model_name in enumerate(model_names):
        sv = all_results[model_name][stage_4]["shap_values"]
        mean_abs = np.abs(sv.values).mean(axis=0)
        row_sum = mean_abs.sum()
        matrix[i] = mean_abs / row_sum if row_sum > 0 else mean_abs

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


def plot_mcao_dependence(all_results: dict[str, dict]) -> None:
    model_names = list(all_results.keys())
    n_cols = 4
    n_rows = (len(model_names) + n_cols - 1) // n_cols
    norm = mcolors.Normalize(vmin=0, vmax=100)
    cmap = shap.plots.colors.red_blue

    for stage_idx, (stage_name, _) in enumerate(STAGED_MODELS, start=1):
        # Collect per-model data for this stage
        plot_data = []
        all_mcao_vals, all_shap_vals = [], []
        for model_name in model_names:
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

        x_lo, x_hi = np.percentile(np.concatenate(all_mcao_vals), [0, 100])
        y_lo, y_hi = np.percentile(np.concatenate(all_shap_vals), [0, 100])

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.5 * n_cols, 4 * n_rows),
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

            if idx >= (n_rows - 1) * n_cols:
                ax.set_xlabel("MCAO (K)", fontsize=12)
            if col == 0:
                ax.set_ylabel("SHAP value for MCAO", fontsize=12)

        for idx in range(len(model_names), n_rows * n_cols):
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


def main() -> None:
    os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)

    all_results = {}
    for model_name in MODELS:
        print(f"=== {model_name} ===")
        all_results[model_name] = run_staged_analysis(model_name)

    print("\nGenerating plots...")
    plot_beeswarms(all_results)
    plot_intermodel_heatmap(all_results)
    plot_mcao_dependence(all_results)

    print("Done.")


if __name__ == "__main__":
    main()
