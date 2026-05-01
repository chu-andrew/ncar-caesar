"""Visualization for the temporal holdout prediction model.

Reads results from run_temporal_analysis and generates:
  - Predicted vs actual scatter (test period 2018-2023)
  - Observable vs all-features R2 comparison bars
  - Residuals by year (temporal drift diagnostic)
  - Summary table
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from swing3.config import MODELS
from swing3.predict import PREDICT_PLOTS_DIR, run_temporal_analysis

_TEST_YEAR_MIN = 2013
_TEST_YEAR_MAX = 2021
_YEAR_CMAP = plt.cm.plasma
_SCENARIO_COLORS = {"observable": "steelblue", "all": "darkorange"}
_SCENARIO_LABELS = {"observable": "Observable (no isotopes)", "all": "All features"}


def _year_colors(years: np.ndarray) -> np.ndarray:
    norm = (years - _TEST_YEAR_MIN) / max(_TEST_YEAR_MAX - _TEST_YEAR_MIN, 1)
    return _YEAR_CMAP(norm)


def plot_pred_vs_actual(
    all_results: dict[str, dict],
    scenario: str = "observable",
) -> None:
    """Scatter of predicted vs actual PE on test set, one panel per model.

    Dots colored by year. 1:1 line shown. Title includes R2 +- std.
    """
    models = [m for m in MODELS if m in all_results and scenario in all_results[m]]
    n = len(models)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    all_vals = np.concatenate([
        arr for m in models
        for arr in (all_results[m][scenario]["y_true_test"], all_results[m][scenario]["y_pred_test"])
    ])
    vmin, vmax = np.nanpercentile(all_vals, [1, 99])

    for ax, model_name in zip(axes.flat, models):
        r = all_results[model_name][scenario]
        colors = _year_colors(r["years_test"])
        ax.scatter(r["y_true_test"], r["y_pred_test"], c=colors, s=20, alpha=0.4, rasterized=True)
        ax.plot([vmin, vmax], [vmin, vmax], "k-", linewidth=0.8, alpha=0.6)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        r2_str = rf"$R^2 = {r['r2_test_mean']:.3f} \pm {r['r2_test_std']:.3f}$"
        ax.set_title(model_name, fontsize=16, pad=6)
        ax.text(0.05, 0.95, r2_str, transform=ax.transAxes, fontsize=12, va="top")
        ax.set_xlabel("Actual PE (%)", fontsize=14)
        ax.set_ylabel("Predicted PE (%)", fontsize=14)
        ax.tick_params(labelsize=12)

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    sm = plt.cm.ScalarMappable(
        cmap=_YEAR_CMAP,
        norm=plt.Normalize(_TEST_YEAR_MIN, _TEST_YEAR_MAX),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.5, pad=0.02, aspect=30)
    cbar.set_label("Test year", fontsize=14)
    cbar.set_ticks(list(range(_TEST_YEAR_MIN, _TEST_YEAR_MAX + 1)))
    cbar.ax.tick_params(labelsize=12)

    label = _SCENARIO_LABELS.get(scenario, scenario)
    fig.suptitle(
        f"Predicted vs Actual PE — {label}\n(test {_TEST_YEAR_MIN}–{_TEST_YEAR_MAX})",
        fontsize=18,
    )

    out = os.path.join(PREDICT_PLOTS_DIR, f"pred_vs_actual_{scenario}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_r2_comparison(all_results: dict[str, dict]) -> None:
    """Paired bar chart: observable vs all-features test R2 per model."""
    models = list(all_results.keys())
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(9, len(models) * 1.4), 6))

    for i, scenario in enumerate(["observable", "all"]):
        means = [
            all_results[m].get(scenario, {}).get("r2_test_mean", float("nan"))
            for m in models
        ]
        stds = [
            all_results[m].get(scenario, {}).get("r2_test_std", float("nan"))
            for m in models
        ]
        offset = (i - 0.5) * width
        ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=4,
            color=_SCENARIO_COLORS[scenario],
            label=_SCENARIO_LABELS[scenario],
            alpha=0.85,
            error_kw={"linewidth": 1.0},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=14)
    ax.set_ylabel(r"Test $R^2$" + f" ({_TEST_YEAR_MIN}–{_TEST_YEAR_MAX})", fontsize=14)
    ax.set_title(
        r"Temporal Holdout $R^2$: Observable vs All Features",
        fontsize=18, pad=12,
    )
    ax.legend(fontsize=14)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = os.path.join(PREDICT_PLOTS_DIR, "r2_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_residuals_by_year(
    all_results: dict[str, dict],
    scenario: str = "observable",
) -> None:
    """Box-and-whisker of (y_true - y_pred) per test year, one panel per model.

    Detects temporal drift in residuals between 2018 and 2023.
    """
    models = [m for m in MODELS if m in all_results and scenario in all_results[m]]
    n = len(models)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 5 * nrows),
        squeeze=False,
        constrained_layout=True,
        sharey=True,
    )

    for ax, model_name in zip(axes.flat, models):
        r = all_results[model_name][scenario]
        residuals = r["y_true_test"] - r["y_pred_test"]
        years = r["years_test"]

        model_years = sorted(np.unique(years).tolist())
        data_by_year = [residuals[years == yr] for yr in model_years]
        ax.boxplot(
            data_by_year,
            tick_labels=model_years,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"facecolor": _SCENARIO_COLORS.get(scenario, "steelblue"), "alpha": 0.6},
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
            whiskerprops={"linewidth": 0.9},
            capprops={"linewidth": 0.9},
        )
        ax.axhline(0, color="red", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.set_title(model_name, fontsize=16, pad=6)
        ax.set_xlabel("Test Year", fontsize=14)
        ax.tick_params(labelsize=12)

    for ax in axes[:, 0]:
        ax.set_ylabel("Residual PE (%)", fontsize=14)

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    label = _SCENARIO_LABELS.get(scenario, scenario)
    fig.suptitle(f"Residuals by Year — {label}", fontsize=18)

    out = os.path.join(PREDICT_PLOTS_DIR, f"residuals_by_year_{scenario}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def print_summary_table(all_results: dict[str, dict], train_cutoff: int = 2017) -> None:
    """Print summary R2 table across models and scenarios.

    r2_test_std reflects model variance across seeds, not test-set sampling variance.
    Effective independent test observations = (2023 - train_cutoff) * 4 JFMA months.
    """
    n_test_years = 2021 - train_cutoff
    n_independent = n_test_years * 4
    print(f"\nTest period: {train_cutoff + 1}–2021 ({n_test_years} years, {n_independent} independent JFMA time points)")
    print("Note: +-std reflects model variance across seeds, not test-set sampling uncertainty.\n")
    print(f"{'Model':<10}  {'Obs feats':>9}  {'Obs R2':>13}  {'All feats':>9}  {'All R2':>13}  {'dR2':>7}")
    print("-" * 72)
    for model_name in MODELS:
        if model_name not in all_results:
            continue
        res = all_results[model_name]
        obs = res.get("observable", {})
        all_ = res.get("all", {})
        n_obs = obs.get("n_features", 0)
        n_all = all_.get("n_features", 0)
        r2_obs = obs.get("r2_test_mean", float("nan"))
        std_obs = obs.get("r2_test_std", float("nan"))
        r2_all = all_.get("r2_test_mean", float("nan"))
        std_all = all_.get("r2_test_std", float("nan"))
        delta = r2_all - r2_obs if not (np.isnan(r2_obs) or np.isnan(r2_all)) else float("nan")
        print(
            f"{model_name:<10}  {n_obs:>9}  {r2_obs:>6.3f} +-{std_obs:.3f}  "
            f"{n_all:>9}  {r2_all:>6.3f} +-{std_all:.3f}  {delta:>+7.3f}"
        )


def main() -> None:
    os.makedirs(PREDICT_PLOTS_DIR, exist_ok=True)

    all_results = {}
    for model_name in MODELS:
        print(f"=== {model_name} ===")
        all_results[model_name] = run_temporal_analysis(model_name)

    print("\nGenerating plots...")
    plot_pred_vs_actual(all_results, scenario="observable")
    plot_pred_vs_actual(all_results, scenario="all")
    plot_r2_comparison(all_results)
    plot_residuals_by_year(all_results, scenario="observable")
    plot_residuals_by_year(all_results, scenario="all")

    print_summary_table(all_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
