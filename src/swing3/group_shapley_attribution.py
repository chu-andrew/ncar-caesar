import os
from itertools import combinations
from math import factorial

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from nc.cache import MEMORY
from nc.loader import PROJECT_ROOT
from swing3.config import (
    GROUP_COLORS,
    GROUP_LABELS,
    MODEL_EXCLUDED_GROUPS,
    MODELS,
    PREDICTOR_GROUPS,
    columns_for_stage,
)
from swing3.features import load_shap_features
from swing3.shap_analysis import _tune_hyperparameters, train_and_explain

PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/remote/swing3/plots/shap/group_shapley")


def _coalition_r2(
    features, target, columns, groups, best_params, random_state=0
) -> float:
    """Return mean test R2 for one coalition, reusing the CV logic in train_and_explain."""
    return train_and_explain(
        features, target, columns, groups, best_params, random_state, compute_shap=False
    )["r2_test_mean"]


def _shapley_from_coalition_r2(coalition_r2: dict, group_names: list) -> dict:
    """
    Apply the Shapley formula over groups to compute group-level Shapley values.

    phi(g) = sum_{S subset G \ {g}} w(s) * [v(S union {g}) - v(S)]

    where w(s) = s! * (n-s-1)! / n! is the probability that a uniformly random
    ordering of groups places g immediately after the s members of S.
    """
    n = len(group_names)
    shapley = {}
    for g in group_names:
        others = [x for x in group_names if x != g]
        value = 0.0
        for s in range(n):  # s = size of coalition S preceding g
            weight = factorial(s) * factorial(n - s - 1) / factorial(n)
            for S in combinations(others, s):
                S_set = frozenset(S)
                value += weight * (coalition_r2[S_set | {g}] - coalition_r2[S_set])
        shapley[g] = value
    return shapley


@MEMORY.cache
def run_group_shapley_attribution(
    model_name: str,
    n_seeds: int = 5,
    isotope_features: tuple[str, ...] | None = None,
    excluded_groups: tuple[str, ...] | None = None,
) -> dict:
    """
    Compute group-level Shapley values for the predictor groups for one model.

    Parameters
    ----------
    isotope_features : tuple[str, ...] | None
        Override the isotope feature list. When None, uses PREDICTOR_GROUPS["isotopes"].
        Pass a tuple (hashable for caching) to run a sensitivity analysis, e.g.
        isotope_features=("dD_gradient", "dexcessp") to exclude dDp.
    excluded_groups : tuple[str, ...] | None
        Groups to exclude entirely from the Shapley game (e.g. ("clouds",) for CAM5
        which has no cloud data). Excluded groups receive a Shapley value of 0.

    Returns a dict
      shapley      :  mean Shapley value per group (0 for excluded groups)
      shapley_std  :  std of Shapley values across seeds (0 for excluded groups)
      coalition_r2 :  mean R2 per coalition (frozenset of group names -> float)
      stage_4_r2   :  R2 of the full model (all active groups)
    """
    print(f"Loading features for {model_name}...")
    features, target, groups = load_shap_features(model_name)

    predictor_groups = dict(PREDICTOR_GROUPS)
    if isotope_features is not None:
        predictor_groups = {**PREDICTOR_GROUPS, "isotopes": list(isotope_features)}

    # Remove excluded groups from the Shapley game
    _excluded = set(excluded_groups) if excluded_groups else set()
    predictor_groups = {g: cols for g, cols in predictor_groups.items()
                        if g not in _excluded}

    group_names = list(predictor_groups.keys())
    n = len(group_names)

    def _cols_for_coalition(coalition):
        return [col for g in sorted(coalition) for col in predictor_groups[g]]

    # all 2^n - 1 non-empty subsets of groups, ordered by size for readable output
    all_coalitions = [
        frozenset(c)
        for size in range(1, n + 1)
        for c in combinations(group_names, size)
    ]

    # evaluate R2 for all coalitions
    # empty coalition = R2 of 0 by definition; all others tuned and run over n_seeds.
    print(f"\tEvaluating {2**n} coalitions ({n_seeds} seeds each)...")
    coalition_r2_seeds: dict[frozenset, list[float]] = {frozenset(): [0.0] * n_seeds}

    for coalition in all_coalitions:
        cols = _cols_for_coalition(coalition)
        print(f"\t\tTuning {set(coalition)}...")
        best_params = _tune_hyperparameters(
            features[cols], target, groups, random_state=10
        )
        r2_seeds: list[float] = Parallel(n_jobs=-1)(
            delayed(_coalition_r2)(features, target, cols, groups, best_params, seed)
            for seed in range(n_seeds)
        )
        coalition_r2_seeds[frozenset(coalition)] = r2_seeds
        print(f"\t\t  R2 = {np.mean(r2_seeds):.3f} +- {np.std(r2_seeds):.3f}")

    # compute Shapley values per seed, then aggregate
    shapley_per_seed = []
    for seed_idx in range(n_seeds):
        r2_this_seed = {k: v[seed_idx] for k, v in coalition_r2_seeds.items()}
        shapley_per_seed.append(_shapley_from_coalition_r2(r2_this_seed, group_names))

    shapley_mean = {
        g: float(np.mean([o[g] for o in shapley_per_seed])) for g in group_names
    }
    shapley_std = {
        g: float(np.std([o[g] for o in shapley_per_seed])) for g in group_names
    }
    # Add zero entries for excluded groups so downstream code sees all groups
    for g in _excluded:
        shapley_mean[g] = 0.0
        shapley_std[g] = 0.0

    coalition_r2_mean = {k: float(np.mean(v)) for k, v in coalition_r2_seeds.items()}
    stage_4_r2 = coalition_r2_mean[frozenset(group_names)]

    # efficiency axiom: active group Shapley values must sum to full-coalition R2
    shapley_sum = sum(shapley_mean[g] for g in group_names)
    assert abs(shapley_sum - stage_4_r2) < 1e-6, (
        f"Efficiency axiom violated for {model_name}: "
        f"sum(Shapley) = {shapley_sum:.6f}, full-coalition R2 = {stage_4_r2:.6f}"
    )

    print(
        f"\tGroup Shapley values (sum = {shapley_sum:.3f}, full-coalition R2 = {stage_4_r2:.3f}):"
    )
    for g in list(group_names) + sorted(_excluded):
        print(f"\t\t{g}: {shapley_mean[g]:.3f} +- {shapley_std[g]:.3f}")

    return {
        "shapley": shapley_mean,
        "shapley_std": shapley_std,
        "coalition_r2": coalition_r2_mean,
        "stage_4_r2": stage_4_r2,
    }


def plot_group_shapley_attribution(all_results: dict[str, dict]) -> None:
    """Stacked bar chart of group Shapley values per model."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    model_names = list(all_results.keys())
    group_names = list(PREDICTOR_GROUPS.keys())
    x = np.arange(len(model_names))
    bottom = np.zeros(len(model_names))

    fig, ax = plt.subplots(figsize=(10, 6))
    for g in group_names:
        vals = np.array([all_results[m]["shapley"][g] for m in model_names])
        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=GROUP_LABELS[g],
            color=GROUP_COLORS[g],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Group Shapley value ($R^2$ attribution)", fontsize=12)
    ax.set_title("Group-level PE attribution via Shapley values", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    out = os.path.join(PLOTS_DIR, "group_shapley_attribution.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def print_comparison_table(all_results: dict[str, dict]) -> None:
    """Print group Shapley values (mean +- std) alongside Stage 4 R2."""
    group_names = list(PREDICTOR_GROUPS.keys())
    col_w = 14
    header = (
        f"{'Model':<10}"
        + "".join(f"  {g:>{col_w}}" for g in group_names)
        + f"  {'Stage4 R2':>10}"
    )
    print(header)
    print("-" * len(header))
    for model_name, result in all_results.items():
        row = f"{model_name:<10}"
        for g in group_names:
            cell = f"{result['shapley'][g]:.3f}+-{result['shapley_std'][g]:.3f}"
            row += f"  {cell:>{col_w}}"
        row += f"  {result['stage_4_r2']:>10.3f}"
        print(row)


def plot_group_fraction_heatmap(all_results: dict[str, dict]) -> None:
    """Heatmap of group Shapley values as a fraction of Stage 4 R2 per model.

    The dominant group per model is shown in bold. Answers: which model is most
    dependent on which predictor group?
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    model_names = list(all_results.keys())
    group_names = list(PREDICTOR_GROUPS.keys())

    matrix = np.zeros((len(model_names), len(group_names)))
    for i, m in enumerate(model_names):
        r2 = all_results[m]["stage_4_r2"]
        for j, g in enumerate(group_names):
            matrix[i, j] = all_results[m]["shapley"][g] / r2 if r2 > 0 else 0

    dominant_idx = matrix.argmax(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(group_names)))
    ax.set_xticklabels([GROUP_LABELS[g] for g in group_names], fontsize=11)
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=11)

    for i in range(len(model_names)):
        for j in range(len(group_names)):
            val = matrix[i, j]
            weight = "bold" if j == dominant_idx[i] else "normal"
            color = "white" if val > 0.5 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=10, fontweight=weight, color=color,
            )

    fig.colorbar(im, ax=ax, label="Fraction of Stage 4 R2", shrink=0.8)
    ax.set_title(
        "Group Shapley fraction of Stage 4 R2 per model\n(bold = dominant group)",
        fontsize=13,
    )

    out = os.path.join(PLOTS_DIR, "group_fraction_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def print_dominant_group_table(all_results: dict[str, dict]) -> None:
    """Print the dominant predictor group per model with its Shapley fraction."""
    group_names = list(PREDICTOR_GROUPS.keys())
    print(f"\n{'Model':<10}  {'Dominant group':<18}  {'Fraction':>8}  {'Stage4 R2':>10}")
    print("-" * 55)
    for model_name, result in all_results.items():
        r2 = result["stage_4_r2"]
        dominant = max(group_names, key=lambda g: result["shapley"][g])
        frac = result["shapley"][dominant] / r2 if r2 > 0 else 0
        print(f"{model_name:<10}  {GROUP_LABELS[dominant]:<18}  {frac:>8.3f}  {r2:>10.3f}")


def print_sensitivity_comparison(
    baseline: dict[str, dict],
    sensitivity: dict[str, dict],
) -> None:
    """Compare isotope group Shapley values with and without dDp."""
    print(f"\n{'Model':<10}  {'Isotopes (full)':>16}  {'Isotopes (no dDp)':>18}  {'Delta':>8}  {'Stage4 R2 (no dDp)':>18}")
    print("-" * 80)
    for model_name in baseline:
        base_iso = baseline[model_name]["shapley"]["isotopes"]
        base_std = baseline[model_name]["shapley_std"]["isotopes"]
        sens_iso = sensitivity[model_name]["shapley"]["isotopes"]
        sens_std = sensitivity[model_name]["shapley_std"]["isotopes"]
        delta = sens_iso - base_iso
        stage4 = sensitivity[model_name]["stage_4_r2"]
        print(
            f"{model_name:<10}  {base_iso:.3f}+-{base_std:.3f}  "
            f"{sens_iso:.3f}+-{sens_std:.3f}  {delta:>+8.3f}  {stage4:>18.3f}"
        )


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    all_results = {}
    for model_name in MODELS:
        print(f"=== {model_name} ===")
        excluded = tuple(sorted(MODEL_EXCLUDED_GROUPS.get(model_name, set())))
        all_results[model_name] = run_group_shapley_attribution(
            model_name, excluded_groups=excluded or None
        )

    print("\nGroup Shapley attribution table:")
    print_comparison_table(all_results)

    print("\nGenerating plots...")
    plot_group_shapley_attribution(all_results)
    plot_group_fraction_heatmap(all_results)

    print("\nDominant group per model:")
    print_dominant_group_table(all_results)

    print("\n=== Sensitivity: dDp excluded from isotope group ===")
    sensitivity_results = {}
    for model_name in MODELS:
        print(f"=== {model_name} ===")
        sensitivity_results[model_name] = run_group_shapley_attribution(
            model_name, isotope_features=("dD_gradient", "dexcessp")
        )

    print("\nSensitivity table (dDp excluded):")
    print_comparison_table(sensitivity_results)

    print("\nIsotope attribution comparison (full vs. no dDp):")
    print_sensitivity_comparison(all_results, sensitivity_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
