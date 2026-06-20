"""Feature-level SHAP analysis: within-group importance, direction consistency."""

import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nc.loader import PROJECT_ROOT
from swing3.config import GROUP_COLORS, GROUP_LABELS, MODELS, PREDICTOR_GROUPS
from swing3.shap_analysis import run_staged_analysis
from swing3.types import StagedResult

SHAP_PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/swing3/plots/shap")


def compute_feature_stats(
    all_results: dict[str, dict[str, StagedResult]],
) -> pd.DataFrame:
    """Mean |SHAP| and Spearman-sign direction (+1/-1/0) per feature x model."""
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

            rows.append(
                {
                    "model": model_name,
                    "feature": feat,
                    "group": feat_to_group[feat],
                    "mean_abs_shap": float(mean_abs[i]),
                    "direction": direction,
                }
            )

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
            vals = [
                model_df.loc[f, "mean_abs_shap"] if f in model_df.index else 0.0
                for f in features
            ]
            offset = (k - n_models / 2 + 0.5) * width
            ax.bar(
                x + offset,
                vals,
                width=width * 0.9,
                color=color,
                label=model_name if group == group_names[0] else None,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=30, ha="right", fontsize=9)
        ax.set_title(GROUP_LABELS[group], fontsize=12, color=GROUP_COLORS[group])
        ax.set_ylabel("Mean |SHAP|" if group == group_names[0] else "")
        ax.grid(True, alpha=0.3, axis="y")

    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper right",
        fontsize=9,
        bbox_to_anchor=(1.0, 1.0),
    )
    fig.suptitle("Within-group feature importance (Stage 4 SHAP)", fontsize=14)
    fig.tight_layout()

    out = os.path.join(SHAP_PLOTS_DIR, "within_group_importance.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_direction_heatmap(feature_stats: pd.DataFrame) -> None:
    """Heatmap of SHAP direction (green=+, red=-) per feature x model; bold = inconsistent."""
    os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)

    group_names = list(PREDICTOR_GROUPS.keys())
    model_names = feature_stats["model"].unique().tolist()

    ordered_features = [f for g in group_names for f in PREDICTOR_GROUPS[g]]
    n_feats = len(ordered_features)
    n_models = len(model_names)

    matrix = np.zeros((n_feats, n_models))
    for j, model_name in enumerate(model_names):
        model_df = feature_stats[feature_stats["model"] == model_name].set_index(
            "feature"
        )
        for i, feat in enumerate(ordered_features):
            if feat in model_df.index:
                matrix[i, j] = model_df.loc[feat, "direction"]

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "dir", ["tab:red", "lightgray", "tab:green"]
    )
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(n_models))
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_yticks(np.arange(n_feats))
    ax.set_yticklabels(ordered_features, fontsize=9)

    for i in range(n_feats):
        row = matrix[i, :]
        consistent = len(set(row[row != 0])) <= 1
        for j in range(n_models):
            val = matrix[i, j]
            symbol = "+" if val > 0 else ("-" if val < 0 else "?")
            weight = "normal" if consistent else "bold"
            ax.text(
                j,
                i,
                symbol,
                ha="center",
                va="center",
                fontsize=11,
                fontweight=weight,
                color="black",
            )

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
            fontsize=9,
            color=GROUP_COLORS[g],
            va="center",
            annotation_clip=False,
        )
        boundary += n

    ax.set_title(
        "SHAP direction per feature x model (Stage 4)\n"
        "bold = inconsistent across models",
        fontsize=12,
    )
    fig.tight_layout()

    out = os.path.join(SHAP_PLOTS_DIR, "direction_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def print_direction_table(feature_stats: pd.DataFrame) -> None:
    """Print direction (+/-) per feature x model with a consistency flag."""
    group_names = list(PREDICTOR_GROUPS.keys())
    model_names = feature_stats["model"].unique().tolist()
    ordered_features = [f for g in group_names for f in PREDICTOR_GROUPS[g]]

    col_w = 7
    header = (
        f"{'Feature':<14}  {'Group':<12}"
        + "".join(f"  {m:>{col_w}}" for m in model_names)
        + "  Consistent"
    )
    print(header)
    print("-" * len(header))

    for feat in ordered_features:
        feat_df = feature_stats[feature_stats["feature"] == feat].set_index("model")
        if feat_df.empty:
            continue
        group = feat_df["group"].iloc[0]
        dirs = [
            feat_df.loc[m, "direction"] if m in feat_df.index else 0
            for m in model_names
        ]
        symbols = ["+" if d > 0 else ("-" if d < 0 else "?") for d in dirs]
        nonzero = [d for d in dirs if d != 0]
        consistent = "YES" if len(set(nonzero)) <= 1 else "NO *"
        row = (
            f"{feat:<14}  {group:<12}"
            + "".join(f"  {s:>{col_w}}" for s in symbols)
            + f"  {consistent}"
        )
        print(row)


def main() -> None:
    os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)

    all_results = {m: run_staged_analysis(m) for m in MODELS}
    feature_stats = compute_feature_stats(all_results)

    plot_within_group_importance(feature_stats)
    plot_direction_heatmap(feature_stats)
    print_direction_table(feature_stats)


if __name__ == "__main__":
    main()
