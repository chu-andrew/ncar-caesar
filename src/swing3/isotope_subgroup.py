import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from nc.cache import MEMORY
from nc.loader import PROJECT_ROOT
from swing3.config import MODELS, PREDICTOR_GROUPS
from swing3.features import load_shap_features
from swing3.group_shapley_attribution import _coalition_r2, _shapley_from_coalition_r2
from swing3.shap_analysis import _tune_hyperparameters

PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/remote/swing3/plots/shap/isotope_subgroup")

_ISOTOPE_FEATURES = PREDICTOR_GROUPS["isotopes"] + ["dDs", "dexcesss"]

_FEATURE_COLORS = {
    "dD_gradient": "#7b2d8b",
    "dDp": "#c06dc8",
    "dexcessp": "#e0b0e8",
    "dDs": "#2d7b8b",
    "dexcesss": "#b0e4e8",
}


@MEMORY.cache
def run_isotope_subgroup_shapley(model_name: str, n_seeds: int = 5) -> dict:
    """
    Compute Shapley values for each isotope feature within the isotope group.

    Treats each of {dD_gradient, dDp, dexcessp} as a player. The Shapley values
    sum to the isotope-only R2 (efficiency axiom). Hyperparameters are tuned once
    on the full 3-feature isotope set; all 2^3 = 8 coalitions are evaluated.

    Returns a dict with keys:
      shapley      : mean Shapley value per feature
      shapley_std  : std across seeds
      coalition_r2 : mean R2 per coalition
      isotope_r2   : R2 of the full isotope model (all 3 features)
    """
    print(f"Loading features for {model_name}...")
    features, target, groups = load_shap_features(model_name)

    feature_names = list(_ISOTOPE_FEATURES)  # dD_gradient, dDp, dexcessp, dDs, dexcesss
    n = len(feature_names)

    all_coalitions = [
        frozenset(c)
        for size in range(1, n + 1)
        for c in combinations(feature_names, size)
    ]

    print(f"\tTuning on full isotope feature set...")
    best_params = _tune_hyperparameters(
        features[feature_names], target, groups, random_state=10
    )

    print(f"\tEvaluating {2**n} coalitions ({n_seeds} seeds each)...")
    coalition_r2_seeds: dict[frozenset, list[float]] = {frozenset(): [0.0] * n_seeds}

    for coalition in all_coalitions:
        cols = sorted(coalition)
        r2_seeds: list[float] = Parallel(n_jobs=-1)(
            delayed(_coalition_r2)(features, target, cols, groups, best_params, seed)
            for seed in range(n_seeds)
        )
        coalition_r2_seeds[frozenset(coalition)] = r2_seeds
        print(f"\t\t{set(coalition)}: R2 = {np.mean(r2_seeds):.3f} +- {np.std(r2_seeds):.3f}")

    shapley_per_seed = []
    for seed_idx in range(n_seeds):
        r2_this_seed = {k: v[seed_idx] for k, v in coalition_r2_seeds.items()}
        shapley_per_seed.append(_shapley_from_coalition_r2(r2_this_seed, feature_names))

    shapley_mean = {
        f: float(np.mean([o[f] for o in shapley_per_seed])) for f in feature_names
    }
    shapley_std = {
        f: float(np.std([o[f] for o in shapley_per_seed])) for f in feature_names
    }
    coalition_r2_mean = {k: float(np.mean(v)) for k, v in coalition_r2_seeds.items()}
    isotope_r2 = coalition_r2_mean[frozenset(feature_names)]

    shapley_sum = sum(shapley_mean.values())
    assert abs(shapley_sum - isotope_r2) < 1e-6, (
        f"Efficiency axiom violated for {model_name}: "
        f"sum(Shapley) = {shapley_sum:.6f}, isotope R2 = {isotope_r2:.6f}"
    )

    print(f"\tIsotope sub-group Shapley (sum = {shapley_sum:.3f}, isotope R2 = {isotope_r2:.3f}):")
    for f in feature_names:
        print(f"\t\t{f}: {shapley_mean[f]:.3f} +- {shapley_std[f]:.3f}")

    return {
        "shapley": shapley_mean,
        "shapley_std": shapley_std,
        "coalition_r2": coalition_r2_mean,
        "isotope_r2": isotope_r2,
    }


def plot_isotope_subgroup_attribution(all_results: dict[str, dict]) -> None:
    """Stacked bar chart of isotope sub-group Shapley values per model."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    model_names = list(all_results.keys())
    feature_names = list(_ISOTOPE_FEATURES)
    x = np.arange(len(model_names))
    bottom = np.zeros(len(model_names))

    fig, ax = plt.subplots(figsize=(10, 6))
    for f in feature_names:
        vals = np.array([all_results[m]["shapley"][f] for m in model_names])
        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=f,
            color=_FEATURE_COLORS[f],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Shapley value ($R^2$ attribution)", fontsize=12)
    ax.set_title("Isotope sub-group Shapley attribution", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    out = os.path.join(PLOTS_DIR, "isotope_subgroup_attribution.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def print_subgroup_table(all_results: dict[str, dict]) -> None:
    """Print isotope sub-group Shapley values (mean +- std) alongside isotope R2."""
    feature_names = list(_ISOTOPE_FEATURES)
    col_w = 20
    header = f"{'Model':<10}" + "".join(f"  {f:>{col_w}}" for f in feature_names) + f"  {'Isotope R2':>10}"
    print(header)
    print("-" * len(header))
    for model_name, result in all_results.items():
        row = f"{model_name:<10}"
        for f in feature_names:
            cell = f"{result['shapley'][f]:.3f}+-{result['shapley_std'][f]:.3f}"
            row += f"  {cell:>{col_w}}"
        row += f"  {result['isotope_r2']:>10.3f}"
        print(row)


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    all_results = {}
    for model_name in MODELS:
        print(f"=== {model_name} ===")
        all_results[model_name] = run_isotope_subgroup_shapley(model_name)

    print("\nIsotope sub-group Shapley table:")
    print_subgroup_table(all_results)

    print("\nGenerating plot...")
    plot_isotope_subgroup_attribution(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
