"""SHAP visualizations for staged PE analysis across WisoMIP models."""

import io
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from nc.loader import PROJECT_ROOT
from swing3.config import (
    GROUP_COLORS,
    GROUP_LABELS,
    MODELS,
    PREDICTOR_GROUPS,
    STAGED_MODELS,
)
from swing3.shap_analysis import (
    run_forward_model,
    run_staged_analysis,
    run_staged_oos_predictions,
    run_surface_isotopes_added,
    run_surface_isotopes_replace,
)

SHAP_PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/remote/swing3/plots/shap")


def plot_beeswarms(all_results: dict[str, dict]) -> None:
    for model_name, stages in all_results.items():
        stage_names = list(stages.keys())
        n = len(stage_names)
        images = []
        for i, stage_name in enumerate(stage_names):
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
    model_names = list(all_results.keys())
    final_stages = {m: list(all_results[m].keys())[-1] for m in model_names}

    # Union of all feature names across models (models may have different feature sets)
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


def plot_mcao_dependence(all_results: dict[str, dict]) -> None:
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


def print_forward_comparison_table(
    staged_results: dict[str, dict],
    forward_results: dict[str, dict],
) -> None:
    """Compare Stage 4 R2 (all features) to forward-only R2 (dDp excluded).

    A small delta indicates dDp is not driving the Stage 4 predictions, which
    addresses the backward-feature concern.
    """
    print(f"\n{'Model':<10}  {'Stage 4 R2':>12}  {'No-dDp R2':>12}  {'Delta':>8}")
    print("-" * 50)
    for model_name in staged_results:
        final_stage = list(staged_results[model_name].keys())[-1]
        r2_full = staged_results[model_name][final_stage]["r2_test_mean"]
        r2_fwd = forward_results[model_name]["r2_test_mean"]
        delta = r2_fwd - r2_full
        print(f"{model_name:<10}  {r2_full:>12.3f}  {r2_fwd:>12.3f}  {delta:>+8.3f}")


def compute_feature_stats(all_results: dict[str, dict]) -> pd.DataFrame:
    """Mean |SHAP| and Spearman-sign direction (+1/-1/0) per feature × model."""
    feat_to_group = {feat: g for g, cols in PREDICTOR_GROUPS.items() for feat in cols}
    rows = []

    for model_name, stages in all_results.items():
        result = stages[list(stages.keys())[-1]]
        sv = result["shap_values"]
        X_full = result["X_full"]
        feature_names = result["feature_names"]
        shap_vals = sv.values
        mean_abs = np.abs(shap_vals).mean(axis=0)

        for i, feat in enumerate(feature_names):
            x = X_full[feat].values
            s = shap_vals[:, i]
            valid = np.isfinite(x) & np.isfinite(s)
            if valid.sum() > 10:
                corr, _ = spearmanr(x[valid], s[valid])
                direction = int(np.sign(corr)) if corr != 0 else 0
            else:
                direction = 0

            rows.append({
                "model": model_name,
                "feature": feat,
                "group": feat_to_group[feat],
                "mean_abs_shap": float(mean_abs[i]),
                "direction": direction,
            })

    return pd.DataFrame(rows)


def plot_within_group_importance(feature_stats: pd.DataFrame) -> None:
    """Grouped bar chart of mean |SHAP| per feature, one subplot per predictor group."""
    os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)

    group_names = list(PREDICTOR_GROUPS.keys())
    model_names = feature_stats["model"].unique().tolist()
    n_models = len(model_names)
    model_colors = plt.cm.tab10(np.linspace(0, 0.9, n_models))

    fig, axes = plt.subplots(1, len(group_names), figsize=(5 * len(group_names), 5))

    for ax, group in zip(axes, group_names):
        group_df = feature_stats[feature_stats["group"] == group]
        features = list(PREDICTOR_GROUPS[group])
        n_feats = len(features)
        width = 0.8 / n_models
        x = np.arange(n_feats)

        for k, (model_name, color) in enumerate(zip(model_names, model_colors)):
            model_df = group_df[group_df["model"] == model_name].set_index("feature")
            vals = [model_df.loc[f, "mean_abs_shap"] if f in model_df.index else 0.0
                    for f in features]
            offset = (k - n_models / 2 + 0.5) * width
            ax.bar(x + offset, vals, width=width * 0.9, color=color,
                   label=model_name if group == group_names[0] else None)

        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=30, ha="right", fontsize=9)
        ax.set_title(GROUP_LABELS[group], fontsize=12, color=GROUP_COLORS[group])
        ax.set_ylabel("Mean |SHAP|" if group == group_names[0] else "")
        ax.grid(True, alpha=0.3, axis="y")

    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper right", fontsize=9, bbox_to_anchor=(1.0, 1.0),
    )
    fig.suptitle("Within-group feature importance (Stage 4 SHAP)", fontsize=14)
    fig.tight_layout()

    out = os.path.join(SHAP_PLOTS_DIR, "within_group_importance.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_direction_heatmap(feature_stats: pd.DataFrame) -> None:
    """Heatmap of SHAP direction (green=+, red=−) per feature × model; bold = inconsistent."""
    os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)

    group_names = list(PREDICTOR_GROUPS.keys())
    model_names = feature_stats["model"].unique().tolist()

    ordered_features = [f for g in group_names for f in PREDICTOR_GROUPS[g]]
    n_feats = len(ordered_features)
    n_models = len(model_names)

    matrix = np.zeros((n_feats, n_models))
    for j, model_name in enumerate(model_names):
        model_df = feature_stats[feature_stats["model"] == model_name].set_index("feature")
        for i, feat in enumerate(ordered_features):
            if feat in model_df.index:
                matrix[i, j] = model_df.loc[feat, "direction"]

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = mcolors.LinearSegmentedColormap.from_list("dir", ["tab:red", "lightgray", "tab:green"])
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(n_models))
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_yticks(np.arange(n_feats))
    ax.set_yticklabels(ordered_features, fontsize=9)

    for i in range(n_feats):
        row = matrix[i, :]
        consistent = len(set(row[row != 0])) <= 1
        for j in range(n_models):
            val = matrix[i, j]
            symbol = "+" if val > 0 else ("−" if val < 0 else "?")
            weight = "normal" if consistent else "bold"
            ax.text(j, i, symbol, ha="center", va="center", fontsize=11,
                    fontweight=weight, color="black")

    boundary = 0
    for g in group_names[:-1]:
        boundary += len(PREDICTOR_GROUPS[g])
        ax.axhline(boundary - 0.5, color="black", linewidth=1.5)

    boundary = 0
    for g in group_names:
        n = len(PREDICTOR_GROUPS[g])
        mid = boundary + n / 2 - 0.5
        ax.annotate(
            GROUP_LABELS[g],
            xy=(1.01, 1 - mid / (n_feats - 1)),
            xycoords=("axes fraction", "axes fraction"),
            fontsize=9, color=GROUP_COLORS[g], va="center",
            annotation_clip=False,
        )
        boundary += n

    ax.set_title(
        "SHAP direction per feature × model (Stage 4)\n"
        "bold = inconsistent across models",
        fontsize=12,
    )
    fig.tight_layout()

    out = os.path.join(SHAP_PLOTS_DIR, "direction_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def print_direction_table(feature_stats: pd.DataFrame) -> None:
    """Print direction (+/-) per feature × model with a consistency flag."""
    group_names = list(PREDICTOR_GROUPS.keys())
    model_names = feature_stats["model"].unique().tolist()
    ordered_features = [f for g in group_names for f in PREDICTOR_GROUPS[g]]

    col_w = 7
    header = f"{'Feature':<14}  {'Group':<12}" + "".join(f"  {m:>{col_w}}" for m in model_names) + "  Consistent"
    print(header)
    print("-" * len(header))

    for feat in ordered_features:
        feat_df = feature_stats[feature_stats["feature"] == feat].set_index("model")
        group = feat_df["group"].iloc[0]
        dirs = [feat_df.loc[m, "direction"] if m in feat_df.index else 0 for m in model_names]
        symbols = ["+" if d > 0 else ("−" if d < 0 else "?") for d in dirs]
        nonzero = [d for d in dirs if d != 0]
        consistent = "YES" if len(set(nonzero)) <= 1 else "NO *"
        row = f"{feat:<14}  {group:<12}" + "".join(f"  {s:>{col_w}}" for s in symbols) + f"  {consistent}"
        print(row)



def plot_r2_by_stage(all_results: dict[str, dict]) -> None:
    """Line plot of test R² at each stage per model, showing marginal gain per predictor group."""
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
        ax.fill_between(xs, means_arr - stds_arr, means_arr + stds_arr, color=color, alpha=0.15)

    ax.set_xticks(x_all)
    ax.set_xticklabels([s.split(": ", 1)[1] if ": " in s else s for s in stage_names], fontsize=10)
    ax.set_ylabel("Mean test R²", fontsize=12)
    ax.set_title("Test R² by stage (marginal gain per predictor group)", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(SHAP_PLOTS_DIR, "r2_by_stage.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_pe_scatter_by_stage(
    staged_results: dict[str, dict],
    oos_results: dict[str, dict[str, dict]],
) -> None:
    """Scatter of climate model PE vs XGBoost OOS prediction, rows=stages, cols=models."""
    os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)

    model_names = list(staged_results.keys())
    stage_names = [name for name, _ in STAGED_MODELS]
    stage_labels = [s.split(": ", 1)[1] if ": " in s else s for s in stage_names]
    n_rows, n_cols = len(stage_names), len(model_names)

    # Shared axis limits across all panels — include both true and predicted
    all_vals = np.concatenate([
        arr
        for m in model_names
        for s in oos_results[m]
        for arr in (oos_results[m][s]["y_true"], oos_results[m][s]["y_oos_pred"])
    ])
    lo, hi = float(np.percentile(all_vals, 1)), float(np.percentile(all_vals, 99))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows),
                             squeeze=False, sharex=True, sharey=True)

    for row, (stage_name, stage_label) in enumerate(zip(stage_names, stage_labels)):
        for col, model_name in enumerate(model_names):
            ax = axes[row, col]

            if stage_name not in oos_results[model_name]:
                ax.set_visible(False)
                continue

            y_true = oos_results[model_name][stage_name]["y_true"]
            y_pred = oos_results[model_name][stage_name]["y_oos_pred"]
            r2 = staged_results[model_name][stage_name]["r2_test_mean"]

            ax.scatter(y_true, y_pred, s=1, alpha=0.4, color="steelblue",
                       linewidths=0, rasterized=True)
            ax.plot([lo, hi], [lo, hi], color="k", linewidth=0.8, linestyle="--")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"$R^2={r2:.2f}$", fontsize=8, pad=2, loc="right")

            if row == 0:
                ax.set_title(model_name, fontsize=12)
                ax.set_title(f"$R^2={r2:.2f}$", fontsize=8, pad=2, loc="right")
            if col == 0:
                ax.set_ylabel(stage_label, fontsize=12)

    fig.supxlabel("Simulated PE (%)", fontsize=14, y=0.01)
    fig.supylabel("Reconstructed PE, held-out (%)", fontsize=14, x=0.01)
    fig.suptitle("Precipitation efficiency: simulated vs. reconstructed", fontsize=18, y=1.01)
    fig.tight_layout()

    out = os.path.join(SHAP_PLOTS_DIR, "pe_scatter_by_stage.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def print_surface_isotope_comparison(
    staged_results: dict[str, dict],
    case_a_results: dict[str, dict],
    case_b_results: dict[str, dict],
) -> None:
    """Compare Stage 4 R² vs surface isotope Case A (+dDs+dexcesss) and Case B (dDp→dDs+dexcesss)."""
    print(
        f"\n{'Model':<10}  {'Stage 4 R2':>12}  {'+dDs+dexcesss':>14}  {'dDp→surface':>12}  "
        f"{'ΔA':>6}  {'ΔB':>6}"
    )
    print("-" * 72)
    for model_name in staged_results:
        final_stage = list(staged_results[model_name].keys())[-1]
        r2_base = staged_results[model_name][final_stage]["r2_test_mean"]
        r2_a = case_a_results[model_name]["r2_test_mean"]
        r2_b = case_b_results[model_name]["r2_test_mean"]
        print(
            f"{model_name:<10}  {r2_base:>12.3f}  {r2_a:>14.3f}  {r2_b:>12.3f}  "
            f"{r2_a - r2_base:>+6.3f}  {r2_b - r2_base:>+6.3f}"
        )


def main() -> None:
    os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)

    all_results = {}
    for model_name in MODELS:
        print(f"=== {model_name} ===")
        all_results[model_name] = run_staged_analysis(model_name)

    print("\nGenerating plots...")
    plot_beeswarms(all_results)
    plot_r2_by_stage(all_results)
    plot_intermodel_heatmap(all_results)
    plot_mcao_dependence(all_results)

    print("\nRunning OOS predictions by stage...")
    oos_results = {m: run_staged_oos_predictions(m) for m in MODELS}
    plot_pe_scatter_by_stage(all_results, oos_results)

    print("\nComputing feature stats...")
    feature_stats = compute_feature_stats(all_results)
    plot_within_group_importance(feature_stats)
    plot_direction_heatmap(feature_stats)

    print("\nDirection consistency table:")
    print_direction_table(feature_stats)

    print("\nRunning forward model (dDp excluded)...")
    forward_results = {m: run_forward_model(m) for m in MODELS}

    print("\nForward model comparison (Stage 4 vs. no-dDp):")
    print_forward_comparison_table(all_results, forward_results)

    print("\nRunning surface isotope comparisons (dDs + dexcesss)...")
    case_a = {m: run_surface_isotopes_added(m) for m in MODELS}
    case_b = {m: run_surface_isotopes_replace(m) for m in MODELS}

    print("\nSurface isotope comparison:")
    print_surface_isotope_comparison(all_results, case_a, case_b)

    print("Done.")


if __name__ == "__main__":
    main()
