"""Isotope sensitivity analysis: dDp circularity check."""

import os

from nc.loader import PROJECT_ROOT
from swing3.config import MODELS
from swing3.shap_analysis import run_forward_model, run_staged_analysis
from swing3.types import StagedResult

SHAP_PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/swing3/plots/shap")


def print_forward_comparison_table(
    staged_results: dict[str, dict[str, StagedResult]],
    forward_results: dict[str, StagedResult],
) -> None:
    """Compare Stage 4 R^2 (all features) to forward-only R^2 (dDp excluded).

    A small delta indicates dDp is not driving the Stage 4 predictions.
    """
    print(f"\n{'Model':<10}  {'Stage 4 R2':>12}  {'No-dDp R2':>12}  {'Delta':>8}")
    print("-" * 50)
    for model_name in staged_results:
        final_stage = list(staged_results[model_name].keys())[-1]
        r2_full = staged_results[model_name][final_stage]["r2_test_mean"]
        r2_fwd = forward_results[model_name]["r2_test_mean"]
        delta = r2_fwd - r2_full
        print(f"{model_name:<10}  {r2_full:>12.3f}  {r2_fwd:>12.3f}  {delta:>+8.3f}")


def main() -> None:
    os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)

    all_results = {m: run_staged_analysis(m) for m in MODELS}
    forward_results = {m: run_forward_model(m) for m in MODELS}

    print_forward_comparison_table(all_results, forward_results)


if __name__ == "__main__":
    main()
