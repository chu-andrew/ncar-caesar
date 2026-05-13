"""Stability-based greedy forward feature selection for PE analysis.

Identifies the minimum set of variables needed to achieve near-optimal R^2.

dDp is excluded from the candidate feature pool: it is co-determined with PE
(both result from the same precipitation event) and should not be used for
atmospheric attribution.
"""

import logging
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from nc.cache import MEMORY
from nc.loader import PROJECT_ROOT
from swing3.config import DEFAULT_CFG, MODELS, ExperimentConfig
from swing3.features import load_shap_features
from swing3.shap_analysis import _tune_hyperparameters, train_and_explain
from swing3.types import (
    ForwardSelectionResult,
    ForwardSelectionStep,
)

logger = logging.getLogger(__name__)

PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/swing3/plots/forward_selection")

# Features excluded from greedy selection regardless of stage availability.
_SELECTION_EXCLUDED = {"dDp"}


@MEMORY.cache
def run_stability_forward_selection(
    model_name: str,
    n_seeds: int = 25,
    cfg: ExperimentConfig = DEFAULT_CFG,
) -> ForwardSelectionResult:
    """Greedy forward selection with stability scoring.

    Phase 1 -- selection: greedily adds features by highest mean test R^2. Hyperparameters
    are tuned once per step on the current selected set (rather than per candidate) so the
    winner isn't chosen partly due to candidate-specific tuning luck.

    Phase 2 -- evaluation: after the selection order is fixed, each prefix is re-evaluated
    with an independent set of seeds (range [n_seeds, 2*n_seeds)) to avoid winner's-curse
    bias in the reported R^2 curve.

    dDp is excluded from the candidate pool (circular predictor co-determined with PE).
    """
    logger.info("[%s] Loading features...", model_name)
    features, target, groups = load_shap_features(model_name)
    all_features = [c for c in features.columns if c not in _SELECTION_EXCLUDED]

    selected: list[str] = []
    remaining: list[str] = list(all_features)
    steps: list[ForwardSelectionStep] = []

    # Phase 1: greedy selection (determines order only, not the reported R^2)
    while remaining:
        k = len(selected) + 1
        tune_cols = selected if selected else all_features
        logger.info(
            "[%s] Step k=%d: tuning on %d cols, %d candidates...",
            model_name,
            k,
            len(tune_cols),
            len(remaining),
        )
        best_params = _tune_hyperparameters(
            features[tune_cols], target, groups, cfg=cfg
        )

        logger.info(
            "[%s] Step k=%d: running %d fits...",
            model_name,
            k,
            len(remaining) * n_seeds,
        )
        pair_results = Parallel(n_jobs=-1)(
            delayed(train_and_explain)(
                features,
                target,
                selected + [candidate],
                groups,
                best_params=best_params,
                cfg=cfg,
                random_state=seed,
                compute_shap=False,
            )
            for candidate in remaining
            for seed in range(n_seeds)
        )

        candidate_r2: dict[str, tuple[float, float]] = {}
        for i, candidate in enumerate(remaining):
            r2_vals = [
                pair_results[i * n_seeds + s]["r2_test_mean"] for s in range(n_seeds)
            ]
            candidate_r2[candidate] = (float(np.mean(r2_vals)), float(np.std(r2_vals)))

        best_feat = max(candidate_r2, key=lambda c: candidate_r2[c][0])
        selected.append(best_feat)
        remaining.remove(best_feat)

        steps.append(
            {
                "k": k,
                "feature": best_feat,
                "selected": list(selected),
                "all_candidate_r2": candidate_r2,
            }
        )
        logger.info(
            "[%s]   selected: %s  (selection R^2=%.3f)",
            model_name,
            best_feat,
            candidate_r2[best_feat][0],
        )

    # Phase 2: independent evaluation of each prefix with fresh seeds
    logger.info(
        "[%s] Phase 2: evaluating %d prefixes independently...",
        model_name,
        len(all_features),
    )
    eval_seeds = range(n_seeds, 2 * n_seeds)
    r2_mean_by_k: dict[int, float] = {}
    r2_std_by_k: dict[int, float] = {}

    for k in range(1, len(all_features) + 1):
        prefix = selected[:k]
        params = _tune_hyperparameters(features[prefix], target, groups, cfg=cfg)
        eval_results = Parallel(n_jobs=-1)(
            delayed(train_and_explain)(
                features,
                target,
                prefix,
                groups,
                best_params=params,
                cfg=cfg,
                random_state=seed,
                compute_shap=False,
            )
            for seed in eval_seeds
        )
        r2_vals = [r["r2_test_mean"] for r in eval_results]
        r2_mean_by_k[k] = float(np.mean(r2_vals))
        r2_std_by_k[k] = float(np.std(r2_vals))
        logger.info(
            "[%s]   k=%d (%s): R^2=%.3f +- %.3f",
            model_name,
            k,
            prefix[-1],
            r2_mean_by_k[k],
            r2_std_by_k[k],
        )

    return {
        "selection_order": selected,
        "r2_mean_by_k": r2_mean_by_k,
        "r2_std_by_k": r2_std_by_k,
        "steps": steps,
    }


def plot_r2_vs_k(
    selection_results: dict[str, ForwardSelectionResult],
) -> None:
    """R^2 vs number of features curve, one line per model."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    model_names = list(selection_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(model_names)))

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, color in zip(model_names, colors):
        res = selection_results[model_name]
        ks = sorted(res["r2_mean_by_k"].keys())
        means = np.array([res["r2_mean_by_k"][k] for k in ks])
        stds = np.array([res["r2_std_by_k"][k] for k in ks])
        ax.plot(ks, means, color=color, label=model_name, linewidth=1.8)
        ax.fill_between(ks, means - stds, means + stds, color=color, alpha=0.15)

    ax.set_xlabel("Number of features (k)", fontsize=12)
    ax.set_ylabel("Mean test R^2", fontsize=12)
    ax.set_title("Forward selection: R^2 vs. number of features", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(PLOTS_DIR, "r2_vs_k.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)


def plot_selection_frequency(
    selection_results: dict[str, ForwardSelectionResult],
) -> None:
    """Heatmap of selection step per feature x model.

    Cell value = step k at which each feature was selected (lower = selected earlier).
    Gray cells indicate a feature absent from that model (e.g. low_cloud in CAM5).
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    model_names = list(selection_results.keys())
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
                ax.text(j, i, "--", ha="center", va="center", fontsize=9, color="gray")
            else:
                ax.text(
                    j,
                    i,
                    str(int(matrix[i, j])),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )

    fig.colorbar(
        im, ax=ax, label="Selection step (lower = selected earlier)", shrink=0.8
    )
    ax.set_title("Feature selection order per model\n(1 = first selected)", fontsize=12)
    fig.tight_layout()

    out = os.path.join(PLOTS_DIR, "selection_order_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)


def print_pareto_table(selection_results: dict[str, ForwardSelectionResult]) -> None:
    """Print mean R^2 across models at each k, with % of full-model R^2."""
    model_names = list(selection_results.keys())
    max_k = max(len(v["selection_order"]) for v in selection_results.values())

    full_r2 = {
        m: selection_results[m]["r2_mean_by_k"][
            len(selection_results[m]["selection_order"])
        ]
        for m in model_names
    }
    mean_full = float(np.mean(list(full_r2.values())))

    print(
        f"\n{'k':>3}  {'Mean R^2':>8}  {'% of full':>10}  {'Features added (majority vote)'}"
    )
    print("-" * 70)

    for k in range(1, max_k + 1):
        models_with_k = [
            m for m in model_names if k in selection_results[m]["r2_mean_by_k"]
        ]
        r2_vals = [selection_results[m]["r2_mean_by_k"][k] for m in models_with_k]
        mean_r2 = float(np.mean(r2_vals))
        pct = 100 * mean_r2 / mean_full if mean_full > 0 else 0

        feats_at_k = [
            selection_results[m]["steps"][k - 1]["feature"] for m in models_with_k
        ]
        top_feat, count = Counter(feats_at_k).most_common(1)[0]
        feat_str = f"{top_feat} ({count}/{len(models_with_k)} models)"

        print(f"{k:>3}  {mean_r2:>8.3f}  {pct:>9.1f}%  {feat_str}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    logger.info("=== Stability forward selection ===")
    selection_results: dict[str, ForwardSelectionResult] = {}
    for model_name in MODELS:
        logger.info("\n--- %s ---", model_name)
        selection_results[model_name] = run_stability_forward_selection(
            model_name, n_seeds=DEFAULT_CFG.n_seeds
        )

    logger.info("\n=== Pareto table ===")
    print_pareto_table(selection_results)

    logger.info("\n=== Generating plots ===")
    plot_r2_vs_k(selection_results)
    plot_selection_frequency(selection_results)

    logger.info("Done.")


if __name__ == "__main__":
    main()
