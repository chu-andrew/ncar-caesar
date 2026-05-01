"""Stability-based greedy forward feature selection for PE analysis.

Identifies the minimum set of variables needed to achieve near-optimal R²,
and evaluates fixed data-source-defined subsets for observational feasibility.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from nc.cache import MEMORY
from nc.loader import PROJECT_ROOT
from swing3.config import MODELS, VARIABLE_DATA_SOURCES
from swing3.features import load_shap_features
from swing3.shap_analysis import _tune_hyperparameters, train_and_explain

PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/remote/swing3/plots/shap/forward_selection")

# Named subsets by data source for observational feasibility evaluation
DATA_SOURCE_SUBSETS: dict[str, list[str]] = {
    "satellite": [v for v, s in VARIABLE_DATA_SOURCES.items() if s == "satellite"],
    "satellite+reanalysis": [
        v for v, s in VARIABLE_DATA_SOURCES.items() if s in {"satellite", "reanalysis"}
    ],
}


@MEMORY.cache
def run_stability_forward_selection(model_name: str, n_seeds: int = 25) -> dict:
    """
    Greedy forward selection with stability scoring.

    Phase 1 — selection: greedily adds features by highest mean test R². Hyperparameters
    are tuned once per step on the current selected set (rather than per candidate) so the
    winner isn't chosen partly due to candidate-specific tuning luck.

    Phase 2 — evaluation: after the selection order is fixed, each prefix is re-evaluated
    with an independent set of seeds (range [n_seeds, 2*n_seeds)) to avoid winner's-curse
    bias in the reported R² curve.

    Returns
    -------
    dict with keys:
      selection_order : list[str]   — features in order selected
      r2_mean_by_k   : dict[int, float]  — from independent Phase 2 evaluation
      r2_std_by_k    : dict[int, float]  — from independent Phase 2 evaluation
      steps          : list[dict]   — per-step detail (feature, selected, all_candidate_r2)
    """
    print(f"[{model_name}] Loading features...")
    features, target, groups = load_shap_features(model_name)
    all_features = list(features.columns)

    selected: list[str] = []
    remaining: list[str] = list(all_features)
    steps: list[dict] = []

    # Phase 1: greedy selection (determines order only, not the reported R²)
    while remaining:
        k = len(selected) + 1
        # Tune once per step on the current selected set; at k=1 tune on the full set
        tune_cols = selected if selected else all_features
        print(f"[{model_name}] Step k={k}: tuning on {len(tune_cols)} cols, {len(remaining)} candidates...")
        best_params = _tune_hyperparameters(features[tune_cols], target, groups)

        print(f"[{model_name}] Step k={k}: running {len(remaining) * n_seeds} fits...")
        pair_results = Parallel(n_jobs=-1)(
            delayed(train_and_explain)(
                features, target, selected + [candidate], groups,
                best_params=best_params,
                random_state=seed, compute_shap=False,
            )
            for candidate in remaining
            for seed in range(n_seeds)
        )

        candidate_r2: dict[str, tuple[float, float]] = {}
        for i, candidate in enumerate(remaining):
            r2_vals = [pair_results[i * n_seeds + s]["r2_test_mean"] for s in range(n_seeds)]
            candidate_r2[candidate] = (float(np.mean(r2_vals)), float(np.std(r2_vals)))

        best_feat = max(candidate_r2, key=lambda c: candidate_r2[c][0])
        selected.append(best_feat)
        remaining.remove(best_feat)

        steps.append({
            "k": k,
            "feature": best_feat,
            "selected": list(selected),
            "all_candidate_r2": candidate_r2,
        })
        print(f"[{model_name}]   selected: {best_feat}  (selection R²={candidate_r2[best_feat][0]:.3f})")

    # Phase 2: independent evaluation of each prefix with fresh seeds
    print(f"[{model_name}] Phase 2: evaluating {len(all_features)} prefixes independently...")
    eval_seeds = range(n_seeds, 2 * n_seeds)
    r2_mean_by_k: dict[int, float] = {}
    r2_std_by_k: dict[int, float] = {}

    for k in range(1, len(all_features) + 1):
        prefix = selected[:k]
        params = _tune_hyperparameters(features[prefix], target, groups)
        eval_results = Parallel(n_jobs=-1)(
            delayed(train_and_explain)(
                features, target, prefix, groups,
                best_params=params,
                random_state=seed, compute_shap=False,
            )
            for seed in eval_seeds
        )
        r2_vals = [r["r2_test_mean"] for r in eval_results]
        r2_mean_by_k[k] = float(np.mean(r2_vals))
        r2_std_by_k[k] = float(np.std(r2_vals))
        print(f"[{model_name}]   k={k} ({prefix[-1]}): R²={r2_mean_by_k[k]:.3f} ± {r2_std_by_k[k]:.3f}")

    return {
        "selection_order": selected,
        "r2_mean_by_k": r2_mean_by_k,
        "r2_std_by_k": r2_std_by_k,
        "steps": steps,
    }


@MEMORY.cache
def run_data_source_subset(model_name: str, subset_name: str, n_runs: int = 25) -> dict:
    """Train and evaluate a fixed data-source-defined subset of variables.

    Parameters
    ----------
    subset_name : one of the keys in DATA_SOURCE_SUBSETS
    """
    columns = DATA_SOURCE_SUBSETS[subset_name]
    features, target, groups = load_shap_features(model_name)
    # Only keep columns present in this model's feature DataFrame
    columns = [c for c in columns if c in features.columns]
    print(f"[{model_name}] Subset '{subset_name}': {len(columns)} features: {columns}")

    best_params = _tune_hyperparameters(features[columns], target, groups)
    run_results = Parallel(n_jobs=-1)(
        delayed(train_and_explain)(
            features, target, columns, groups,
            best_params=best_params, random_state=seed, compute_shap=False,
        )
        for seed in range(n_runs)
    )
    r2_vals = [r["r2_test_mean"] for r in run_results]
    result = {
        "r2_test_mean": float(np.mean(r2_vals)),
        "r2_test_std": float(np.std(r2_vals)),
        "columns": columns,
        "n_features": len(columns),
    }
    print(f"[{model_name}]   R²={result['r2_test_mean']:.3f} ± {result['r2_test_std']:.3f}")
    return result


def plot_r2_vs_k(
    selection_results: dict[str, dict],
    subset_results: dict[str, dict[str, dict]],
) -> None:
    """R² vs number of features curve, one line per model.

    Overlays labeled markers for each DATA_SOURCE_SUBSETS at their k position
    (the k at which the subset's R² would first be achieved by forward selection).

    Parameters
    ----------
    selection_results : model_name → run_stability_forward_selection result
    subset_results    : subset_name → {model_name → run_data_source_subset result}
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    model_names = list(selection_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(model_names)))

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, color in zip(model_names, colors):
        res = selection_results[model_name]
        ks = sorted(res["r2_mean_by_k"].keys())
        means = [res["r2_mean_by_k"][k] for k in ks]
        stds = [res["r2_std_by_k"][k] for k in ks]
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.plot(ks, means_arr, color=color, label=model_name, linewidth=1.8)
        ax.fill_between(ks, means_arr - stds_arr, means_arr + stds_arr,
                        color=color, alpha=0.15)

    # Overlay subset markers (averaged across models)
    subset_markers = {"satellite": "^", "satellite+reanalysis": "s"}
    for subset_name, marker in subset_markers.items():
        if subset_name not in subset_results:
            continue
        sub = subset_results[subset_name]
        # Find k for each model: first k where forward selection R² ≥ subset R²
        k_vals, r2_vals = [], []
        for model_name in model_names:
            if model_name not in sub:
                continue
            subset_r2 = sub[model_name]["r2_test_mean"]
            res = selection_results[model_name]
            ks = sorted(res["r2_mean_by_k"].keys())
            k_match = next(
                (k for k in ks if res["r2_mean_by_k"][k] >= subset_r2),
                ks[-1],
            )
            k_vals.append(k_match)
            r2_vals.append(subset_r2)
        if k_vals:
            ax.scatter(
                np.mean(k_vals), np.mean(r2_vals),
                marker=marker, s=100, zorder=5, color="black",
                label=f"{subset_name} (avg across models)",
            )

    ax.set_xlabel("Number of features (k)", fontsize=12)
    ax.set_ylabel("Mean test R²", fontsize=12)
    ax.set_title("Forward selection: R² vs. number of features", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(PLOTS_DIR, "r2_vs_k.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_selection_frequency(selection_results: dict[str, dict]) -> None:
    """Heatmap of selection step per feature × model.

    Cell value = step k at which each feature was selected (lower = selected earlier).
    Gray cells indicate a feature absent from that model (e.g. low_cloud in CAM5).
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    model_names = list(selection_results.keys())
    # Union of all features in insertion order so no model's features are dropped
    seen: set[str] = set()
    all_features: list[str] = []
    for m in model_names:
        for f in selection_results[m]["selection_order"]:
            if f not in seen:
                seen.add(f)
                all_features.append(f)

    max_k = max(len(selection_results[m]["selection_order"]) for m in model_names)
    matrix = np.full((len(all_features), len(model_names)), np.nan)
    for j, model_name in enumerate(model_names):
        order = selection_results[model_name]["selection_order"]
        for feat in order:
            matrix[all_features.index(feat), j] = order.index(feat) + 1

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad("lightgray")
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=1, vmax=max_k)
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_yticks(np.arange(len(all_features)))
    ax.set_yticklabels(all_features, fontsize=9)

    for i in range(len(all_features)):
        for j in range(len(model_names)):
            if np.isnan(matrix[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="gray")
            else:
                ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center",
                        fontsize=9, color="black")

    fig.colorbar(im, ax=ax, label="Selection step (lower = selected earlier)", shrink=0.8)
    ax.set_title("Feature selection order per model\n(1 = first selected)", fontsize=12)
    fig.tight_layout()

    out = os.path.join(PLOTS_DIR, "selection_order_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def print_pareto_table(selection_results: dict[str, dict]) -> None:
    """Print mean R² across models at each k, with % of full-model R²."""
    from collections import Counter

    model_names = list(selection_results.keys())
    max_k = max(len(v["selection_order"]) for v in selection_results.values())

    # Full-model R² per model at their respective final k
    full_r2 = {
        m: selection_results[m]["r2_mean_by_k"][len(selection_results[m]["selection_order"])]
        for m in model_names
    }
    mean_full = np.mean(list(full_r2.values()))

    print(f"\n{'k':>3}  {'Mean R²':>8}  {'% of full':>10}  {'Features added (majority vote)':}")
    print("-" * 70)

    for k in range(1, max_k + 1):
        # Only average over models that have a k-th step
        models_with_k = [m for m in model_names if k in selection_results[m]["r2_mean_by_k"]]
        r2_vals = [selection_results[m]["r2_mean_by_k"][k] for m in models_with_k]
        mean_r2 = np.mean(r2_vals)
        pct = 100 * mean_r2 / mean_full if mean_full > 0 else 0

        feats_at_k = [selection_results[m]["steps"][k - 1]["feature"] for m in models_with_k]
        top_feat, count = Counter(feats_at_k).most_common(1)[0]
        feat_str = f"{top_feat} ({count}/{len(models_with_k)} models)"

        print(f"{k:>3}  {mean_r2:>8.3f}  {pct:>9.1f}%  {feat_str}")


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=== Stability forward selection ===")
    selection_results = {}
    for model_name in MODELS:
        print(f"\n--- {model_name} ---")
        selection_results[model_name] = run_stability_forward_selection(model_name, n_seeds=5)

    print("\n=== Data source subset evaluation ===")
    subset_results: dict[str, dict[str, dict]] = {}
    for subset_name in DATA_SOURCE_SUBSETS:
        print(f"\n-- Subset: {subset_name} --")
        subset_results[subset_name] = {}
        for model_name in MODELS:
            subset_results[subset_name][model_name] = run_data_source_subset(
                model_name, subset_name, n_runs=5
            )

    print("\n=== Pareto table ===")
    print_pareto_table(selection_results)

    print("\n=== Data source subset R² ===")
    print(f"\n{'Model':<10}", end="")
    for subset_name in DATA_SOURCE_SUBSETS:
        print(f"  {subset_name:>22}", end="")
    print()
    print("-" * (10 + 25 * len(DATA_SOURCE_SUBSETS)))
    for model_name in MODELS:
        print(f"{model_name:<10}", end="")
        for subset_name in DATA_SOURCE_SUBSETS:
            r = subset_results[subset_name][model_name]
            print(f"  {r['r2_test_mean']:>22.3f}", end="")
        print()

    print("\n=== Generating plots ===")
    plot_r2_vs_k(selection_results, subset_results)
    plot_selection_frequency(selection_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
