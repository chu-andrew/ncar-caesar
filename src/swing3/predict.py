"""Temporal holdout prediction model for precipitation efficiency (PE).

Trains on GCM data (1979-2012) and evaluates on a held-out period (2013-2021).
Uses all features selected by forward selection at a 95% R^2 threshold.

dDp is excluded from feature selection inputs (circular predictor; see
forward_selection.py for rationale).
"""

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit

from nc.cache import MEMORY
from nc.loader import PROJECT_ROOT
from swing3.config import DEFAULT_CFG, MODELS, ExperimentConfig
from swing3.features import load_predict_features
from swing3.forward_selection import run_stability_forward_selection
from swing3.shap_analysis import _tune_hyperparameters
from swing3.types import TemporalResult, TemporalRunResult

logger = logging.getLogger(__name__)

PREDICT_PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/swing3/plots/predict")


def get_selected_features(
    model_name: str,
    r2_threshold: float = 0.95,
    n_seeds: int = 5,
) -> list[str]:
    """Return selected features from forward selection at r2_threshold.

    Finds the smallest k such that r2_mean_by_k[k] >= r2_threshold * max_r2.
    dDp is never present (excluded upstream in forward selection).
    """
    result = run_stability_forward_selection(model_name, n_seeds=n_seeds)
    selection_order = result["selection_order"]
    r2_by_k = result["r2_mean_by_k"]

    max_r2 = max(r2_by_k.values())
    cutoff_r2 = r2_threshold * max_r2

    k_cutoff = max(r2_by_k)
    for k in sorted(r2_by_k):
        if r2_by_k[k] >= cutoff_r2:
            k_cutoff = k
            break

    return selection_order[:k_cutoff]


def temporal_split(
    years: np.ndarray,
    train_cutoff: int = 2012,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean (train_mask, test_mask) based on calendar year."""
    return years <= train_cutoff, years > train_cutoff


def _train_one_seed(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    years_test: np.ndarray,
    best_params: dict[str, Any],
    cfg: ExperimentConfig = DEFAULT_CFG,
    seed: int = 0,
) -> TemporalRunResult:
    """Train on training data with inner val fold for early stopping; evaluate on test."""
    inner_gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, val_idx = next(inner_gss.split(X_train, y_train, groups_train))

    model = xgb.XGBRegressor(
        n_estimators=cfg.n_estimators,
        tree_method="hist",
        random_state=seed,
        early_stopping_rounds=cfg.early_stopping_rounds,
        **best_params,
    )
    model.fit(
        X_train.iloc[tr_idx],
        y_train[tr_idx],
        eval_set=[(X_train.iloc[val_idx], y_train[val_idx])],
        verbose=False,
    )

    y_pred_test = model.predict(X_test).clip(cfg.pe_min, cfg.pe_max)

    return {
        "r2_train": float(model.score(X_train.iloc[tr_idx], y_train[tr_idx])),
        "r2_test": float(model.score(X_test, y_test)),
        "y_pred_test": y_pred_test,
        "y_true_test": y_test,
        "years_test": years_test,
    }


def _aggregate_temporal_runs(
    run_results: list[TemporalRunResult],
    features_used: list[str],
) -> TemporalResult:
    """Average N seed results into ensemble metrics and mean predictions."""
    r2_inner_list = [r["r2_train"] for r in run_results]
    r2_test_list = [r["r2_test"] for r in run_results]
    y_pred_test_mean = np.stack([r["y_pred_test"] for r in run_results]).mean(axis=0)

    return {
        "r2_inner_mean": float(np.mean(r2_inner_list)),
        "r2_inner_std": float(np.std(r2_inner_list)),
        "r2_test_mean": float(np.mean(r2_test_list)),
        "r2_test_std": float(np.std(r2_test_list)),
        "y_pred_test": y_pred_test_mean,
        "y_true_test": run_results[0]["y_true_test"],
        "years_test": run_results[0]["years_test"],
        "r2_per_seed": r2_test_list,
        "features_used": features_used,
        "n_features": len(features_used),
        "n_runs": len(run_results),
    }


@MEMORY.cache
def run_temporal_analysis(
    model_name: str,
    r2_threshold: float = 0.95,
    n_runs: int = 25,
    train_cutoff: int = 2012,
    cfg: ExperimentConfig = DEFAULT_CFG,
) -> TemporalResult:
    """Temporal holdout prediction for one climate model.

    Returns aggregated metrics and predictions for the test period using all
    forward-selected features. dDp is never present (excluded in forward selection).
    """
    features, target, groups, years = load_predict_features(model_name)

    if years.min() != 1979:
        raise ValueError(
            f"[{model_name}] Unexpected start year: {years.min()}, expected 1979"
        )
    if years.max() <= train_cutoff:
        raise ValueError(f"[{model_name}] No test data: all years <= {train_cutoff}")

    train_mask, test_mask = temporal_split(years, train_cutoff)
    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())
    n_test_years = int(years[test_mask].max() - train_cutoff)
    n_independent = n_test_years * 4
    logger.info(
        "[%s] %d samples -- train=%d (<=%d), test=%d (>%d, %d independent time points)",
        model_name,
        len(features),
        n_train,
        train_cutoff,
        n_test,
        train_cutoff,
        n_independent,
    )

    columns = get_selected_features(model_name, r2_threshold)
    logger.info(
        "[%s] Forward selection @ %.0f%%: %d features: %s",
        model_name,
        100 * r2_threshold,
        len(columns),
        columns,
    )

    X = features[columns]
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = target[train_mask]
    y_test = target[test_mask]
    groups_train = groups[train_mask]
    years_test = years[test_mask]

    logger.info("[%s] Tuning hyperparameters on training data...", model_name)
    best_params = _tune_hyperparameters(X_train, y_train, groups_train, cfg=cfg)

    logger.info("[%s] Running %d seeds...", model_name, n_runs)
    run_results: list[TemporalRunResult] = Parallel(n_jobs=-1)(  # type: ignore[assignment]
        delayed(_train_one_seed)(
            X_train,
            y_train,
            groups_train,
            X_test,
            y_test,
            years_test,
            best_params,
            cfg,
            s,
        )
        for s in range(n_runs)
    )
    result = _aggregate_temporal_runs(run_results, columns)
    logger.info(
        "[%s] R^2 inner=%.3f (+-%.3f), test=%.3f (+-%.3f)",
        model_name,
        result["r2_inner_mean"],
        result["r2_inner_std"],
        result["r2_test_mean"],
        result["r2_test_std"],
    )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(PREDICT_PLOTS_DIR, exist_ok=True)

    all_results: dict[str, TemporalResult] = {}
    for model_name in MODELS:
        logger.info("\n=== %s ===", model_name)
        all_results[model_name] = run_temporal_analysis(model_name)

    print("\n=== Summary ===")
    print(f"{'Model':<10}  {'Features':>8}  {'Test R^2':>10}")
    print("-" * 35)
    for model_name, res in all_results.items():
        print(
            f"{model_name:<10}  {res['n_features']:>8}  "
            f"{res['r2_test_mean']:>6.3f} +-{res['r2_test_std']:.3f}"
        )


if __name__ == "__main__":
    main()
