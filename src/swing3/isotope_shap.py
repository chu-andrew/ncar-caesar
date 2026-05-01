import os

import matplotlib.pyplot as plt
import shap
from joblib import Parallel, delayed

from nc.cache import MEMORY
from nc.loader import PROJECT_ROOT
from swing3.config import MODELS, PREDICTOR_GROUPS
from swing3.features import load_shap_features
from swing3.shap_analysis import (
    _aggregate_runs,
    _tune_hyperparameters,
    train_and_explain,
)

SHAP_PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/remote/swing3/plots/shap/isotope")


@MEMORY.cache
def run_isotope_only_analysis(model_name: str, n_runs: int = 25) -> dict:
    """
    Train and explain an isotope-only model for one climate model, averaged over n_runs seeds.
    """
    features, target, groups = load_shap_features(model_name)
    columns = PREDICTOR_GROUPS["isotopes"]

    print("\t\tTuning hyperparameters...")
    best_params = _tune_hyperparameters(features[columns], target, groups)
    print(
        f"\t\tBest params: { {k: round(v, 3) if isinstance(v, float) else v for k, v in best_params.items()} }"
    )

    run_results = Parallel(n_jobs=-1)(
        delayed(train_and_explain)(
            features,
            target,
            columns,
            groups,
            best_params=best_params,
            random_state=seed,
        )
        for seed in range(n_runs)
    )
    result = _aggregate_runs(run_results)
    print(
        f"\t\tR2 train={result['r2_train_mean']:.3f} (+-{result['r2_train_std']:.3f}), "
        f"test={result['r2_test_mean']:.3f} (+-{result['r2_test_std']:.3f})"
    )
    return result


def print_comparison_table(
    all_staged: dict[str, dict], all_isotope: dict[str, dict]
) -> None:
    """Print isotope-only R2 alongside the penultimate and final staged R2 per model.

    The penultimate stage is the last stage before isotopes were added; the final
    stage is the full model including isotopes. For models without a clouds stage
    (e.g. CAM5) these will differ from other models.
    """
    header = f"{'Model':<10}  {'Isotope-only':>12}  {'Pre-isotope':>12}  {'Final':>8}  {'Delta':>8}"
    print(header)
    print("-" * len(header))
    for model_name in MODELS:
        stage_keys = list(all_staged[model_name].keys())
        iso = all_isotope[model_name]["r2_test_mean"]
        s_pre = all_staged[model_name][stage_keys[-2]]["r2_test_mean"]
        s_final = all_staged[model_name][stage_keys[-1]]["r2_test_mean"]
        print(
            f"{model_name:<10}  {iso:>12.3f}  {s_pre:>12.3f}  {s_final:>8.3f}  {s_final - s_pre:>+8.3f}"
        )


def plot_isotope_beeswarms(all_isotope: dict[str, dict]) -> None:
    """Save one beeswarm plot per model for the isotope-only model."""
    for model_name, result in all_isotope.items():
        r2 = result["r2_test_mean"]

        with plt.rc_context({"font.size": 14}):
            shap.plots.beeswarm(result["shap_values"], show=False, max_display=3)
            fig = plt.gcf()
            fig.suptitle(f"{model_name} (Isotope-only)  ($R^2={r2:.3f}$)", fontsize=14)

        out = os.path.join(SHAP_PLOTS_DIR, f"{model_name}_isotope_beeswarm.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


def main() -> None:
    from swing3.shap_analysis import run_staged_analysis

    os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)

    all_isotope = {}
    for model_name in MODELS:
        print(f"=== {model_name} ===")
        all_isotope[model_name] = run_isotope_only_analysis(model_name)

    print("\nLoading staged results for comparison...")
    all_staged = {m: run_staged_analysis(m) for m in MODELS}

    print("\nComparison table:")
    print_comparison_table(all_staged, all_isotope)

    print("\nGenerating beeswarm plots...")
    plot_isotope_beeswarms(all_isotope)

    print("\nDone.")


if __name__ == "__main__":
    main()
