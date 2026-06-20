"""XGBoost training, Optuna HPO, SHAP computation for staged PE analysis."""

import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit

from nc.cache import MEMORY
from swing3.config import (
    DEFAULT_CFG,
    ExperimentConfig,
    columns_for_stage,
    stages_for_model,
)
from swing3.features import load_shap_features
from swing3.types import OOSResult, RunResult, RunResultWithShap, StagedResult

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def _tune_hyperparameters(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: ExperimentConfig = DEFAULT_CFG,
    random_state: int = 0,
) -> dict[str, Any]:
    """Run an Optuna TPE study on one GroupShuffleSplit fold and return best params."""
    gss = GroupShuffleSplit(
        n_splits=1, test_size=cfg.hparam_val_size, random_state=random_state
    )
    train_idx, _ = next(gss.split(X, y, groups))
    X_tune, y_tune, g_tune = X.iloc[train_idx], y[train_idx], groups[train_idx]

    inner_cv = GroupShuffleSplit(
        n_splits=3, test_size=cfg.hparam_val_size, random_state=random_state
    )

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.16),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0),
        }
        r2_scores: list[float] = []
        for tr_idx, val_idx in inner_cv.split(X_tune, y_tune, g_tune):
            m = xgb.XGBRegressor(
                n_estimators=cfg.n_estimators,
                tree_method="hist",
                random_state=random_state,
                early_stopping_rounds=cfg.early_stopping_rounds,
                **params,
            )
            m.fit(
                X_tune.iloc[tr_idx],
                y_tune[tr_idx],
                eval_set=[(X_tune.iloc[val_idx], y_tune[val_idx])],
                verbose=False,
            )
            r2_scores.append(float(m.score(X_tune.iloc[val_idx], y_tune[val_idx])))
        return float(np.mean(r2_scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=cfg.n_optuna_trials, show_progress_bar=False)
    return study.best_params


def train_and_explain(
    features: pd.DataFrame,
    target: np.ndarray,
    columns: list[str],
    groups: np.ndarray,
    best_params: dict[str, Any],
    cfg: ExperimentConfig = DEFAULT_CFG,
    random_state: int = 0,
    compute_shap: bool = True,
) -> RunResult | RunResultWithShap:
    """Train XGBoost across n_folds outer CV folds and optionally compute SHAP values.

    Each split is 3-way (train / val / test): val is carved from train and used
    solely for early stopping. R^2 is evaluated on the held-out test set across all
    outer folds. The model from the first fold is used for SHAP (when compute_shap=True).

    When compute_shap=False, only R^2 statistics are returned. Used by
    group_shapley_attribution and forward_selection to evaluate coalition R^2 without
    the overhead of SHAP computation.
    """
    X = features[columns]
    y = target

    model = xgb.XGBRegressor(
        n_estimators=cfg.n_estimators,
        tree_method="hist",
        random_state=random_state,
        early_stopping_rounds=cfg.early_stopping_rounds,
        **best_params,
    )

    outer_gss = GroupShuffleSplit(
        n_splits=cfg.n_folds, test_size=0.3, random_state=random_state
    )
    inner_gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)

    r2_train_list: list[float] = []
    r2_test_list: list[float] = []
    shap_tr_idx: np.ndarray | None = None
    shap_val_idx: np.ndarray | None = None

    for i, (outer_train_idx, test_idx) in enumerate(outer_gss.split(X, y, groups)):
        inner_tr, inner_val = next(
            inner_gss.split(
                X.iloc[outer_train_idx], y[outer_train_idx], groups[outer_train_idx]
            )
        )
        tr_idx = outer_train_idx[inner_tr]
        val_idx = outer_train_idx[inner_val]

        if i == 0:
            shap_tr_idx, shap_val_idx = tr_idx, val_idx

        m_cv = xgb.XGBRegressor(**model.get_params())
        m_cv.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(X.iloc[val_idx], y[val_idx])],
            verbose=False,
        )
        r2_train_list.append(float(m_cv.score(X.iloc[tr_idx], y[tr_idx])))
        r2_test_list.append(float(m_cv.score(X.iloc[test_idx], y[test_idx])))

    result: RunResult = {
        "r2_train_mean": float(np.mean(r2_train_list)),
        "r2_test_mean": float(np.mean(r2_test_list)),
        "r2_train_std": float(np.std(r2_train_list)),
        "r2_test_std": float(np.std(r2_test_list)),
        "feature_names": columns,
        "best_params": best_params,
    }

    if compute_shap:
        assert shap_tr_idx is not None and shap_val_idx is not None
        model.fit(
            X.iloc[shap_tr_idx],
            y[shap_tr_idx],
            eval_set=[(X.iloc[shap_val_idx], y[shap_val_idx])],
            verbose=False,
        )
        shap_values = shap.TreeExplainer(model)(X)
        y_pred = model.predict(X).clip(cfg.pe_min, cfg.pe_max)
        shap_result: RunResultWithShap = {
            **result,  # type: ignore[misc]
            "shap_values": shap_values,
            "X_full": X,
            "y_full": y,
            "residuals": y - y_pred,
        }
        return shap_result

    return result


def _aggregate_runs(run_results: list[RunResultWithShap]) -> StagedResult:
    """Average N train_and_explain results into one ensemble result."""
    first = run_results[0]
    shap_vals = np.mean([r["shap_values"].values for r in run_results], axis=0)
    base_val = float(
        np.mean([np.mean(r["shap_values"].base_values) for r in run_results])
    )
    residuals = np.mean([r["residuals"] for r in run_results], axis=0)

    return {
        "shap_values": shap.Explanation(
            values=shap_vals,
            base_values=base_val,
            data=first["X_full"].values,
            feature_names=first["feature_names"],
        ),
        "X_full": first["X_full"],
        "y_full": first["y_full"],
        "r2_train_mean": float(np.mean([r["r2_train_mean"] for r in run_results])),
        "r2_test_mean": float(np.mean([r["r2_test_mean"] for r in run_results])),
        "r2_train_std": float(np.std([r["r2_train_mean"] for r in run_results])),
        "r2_test_std": float(np.std([r["r2_test_mean"] for r in run_results])),
        "feature_names": first["feature_names"],
        "residuals": residuals,
        "n_runs": len(run_results),
    }


@MEMORY.cache
def run_staged_analysis(
    model_name: str,
    n_runs: int = 25,
    cfg: ExperimentConfig = DEFAULT_CFG,
) -> dict[str, StagedResult]:
    """Run all staged SHAP analyses for one climate model, averaged over n_runs seeds."""
    logger.info("Loading features for %s...", model_name)
    features, target, groups = load_shap_features(model_name)
    logger.info("%s: %d samples", model_name, len(features))

    results: dict[str, StagedResult] = {}
    for stage_name, group_keys in stages_for_model(model_name):
        columns = columns_for_stage(group_keys)

        logger.info(
            "\t%s (%d predictors, %d runs)...", stage_name, len(columns), n_runs
        )
        logger.info("\t\tTuning hyperparameters...")
        best_params = _tune_hyperparameters(features[columns], target, groups, cfg=cfg)
        logger.info(
            "\t\tBest params: %s",
            {
                k: round(v, 3) if isinstance(v, float) else v
                for k, v in best_params.items()
            },
        )

        run_results: list[RunResultWithShap] = Parallel(n_jobs=-1)(  # type: ignore[assignment]
            delayed(train_and_explain)(
                features,
                target,
                columns,
                groups,
                best_params=best_params,
                cfg=cfg,
                random_state=seed,
            )
            for seed in range(n_runs)
        )
        results[stage_name] = _aggregate_runs(run_results)
        r = results[stage_name]
        logger.info(
            "\t\tR^2 train=%.3f (+-%.3f), test=%.3f (+-%.3f)",
            r["r2_train_mean"],
            r["r2_train_std"],
            r["r2_test_mean"],
            r["r2_test_std"],
        )

    return results


@MEMORY.cache
def run_forward_model(
    model_name: str,
    n_runs: int = 25,
    cfg: ExperimentConfig = DEFAULT_CFG,
) -> StagedResult:
    """Stage 4 model with dDp excluded; tests whether dDp drives the attribution."""
    columns = [
        c for c in columns_for_stage(_final_stage_groups(model_name)) if c != "dDp"
    ]
    return _run_isotope_variant(model_name, columns, "dDp excluded", n_runs, cfg)


def _run_isotope_variant(
    model_name: str,
    columns: list[str],
    label: str,
    n_runs: int,
    cfg: ExperimentConfig = DEFAULT_CFG,
) -> StagedResult:
    """Shared implementation for surface isotope comparison runs."""
    logger.info("Loading features for %s...", model_name)
    features, target, groups = load_shap_features(model_name)
    logger.info(
        "%s: %d samples, %d features (%s)",
        model_name,
        len(features),
        len(columns),
        label,
    )

    logger.info("\t\tTuning hyperparameters...")
    best_params = _tune_hyperparameters(features[columns], target, groups, cfg=cfg)

    run_results: list[RunResultWithShap] = Parallel(n_jobs=-1)(  # type: ignore[assignment]
        delayed(train_and_explain)(
            features,
            target,
            columns,
            groups,
            best_params=best_params,
            cfg=cfg,
            random_state=seed,
        )
        for seed in range(n_runs)
    )
    result = _aggregate_runs(run_results)
    logger.info(
        "\t\tR^2 train=%.3f (+-%.3f), test=%.3f (+-%.3f)",
        result["r2_train_mean"],
        result["r2_train_std"],
        result["r2_test_mean"],
        result["r2_test_std"],
    )
    return result


def _final_stage_groups(model_name: str) -> list[str]:
    """Return the group keys for this model's final stage."""
    return stages_for_model(model_name)[-1][1]


def _collect_oos_predictions(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    best_params: dict[str, Any],
    cfg: ExperimentConfig = DEFAULT_CFG,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run n_folds-fold CV and return (pred_sum, pred_count) for all points.

    Returns raw accumulators rather than a divided array so the caller can
    aggregate across seeds before dividing, avoiding bias from points that
    happen to fall outside all test folds in a given seed.
    """
    model = xgb.XGBRegressor(
        n_estimators=cfg.n_estimators,
        tree_method="hist",
        random_state=random_state,
        early_stopping_rounds=cfg.early_stopping_rounds,
        **best_params,
    )
    outer_gss = GroupShuffleSplit(
        n_splits=cfg.n_folds, test_size=0.3, random_state=random_state
    )
    inner_gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)

    pred_sum = np.zeros(len(y))
    pred_count = np.zeros(len(y))

    for outer_train_idx, test_idx in outer_gss.split(X, y, groups):
        inner_tr, inner_val = next(
            inner_gss.split(
                X.iloc[outer_train_idx], y[outer_train_idx], groups[outer_train_idx]
            )
        )
        tr_idx = outer_train_idx[inner_tr]
        val_idx = outer_train_idx[inner_val]
        m = xgb.XGBRegressor(**model.get_params())
        m.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(X.iloc[val_idx], y[val_idx])],
            verbose=False,
        )
        pred_sum[test_idx] += m.predict(X.iloc[test_idx])
        pred_count[test_idx] += 1

    return pred_sum, pred_count


@MEMORY.cache
def run_staged_oos_predictions(
    model_name: str,
    n_runs: int = 25,
    cfg: ExperimentConfig = DEFAULT_CFG,
) -> dict[str, OOSResult]:
    """OOS predictions for each stage, averaged over n_runs seeds.

    Separate from run_staged_analysis so the SHAP cache is not disturbed.
    Returns: stage_name -> {"y_true": ..., "y_oos_pred": ...}
    """
    features, target, groups = load_shap_features(model_name)

    results: dict[str, dict[str, np.ndarray]] = {}
    for stage_name, group_keys in stages_for_model(model_name):
        columns = columns_for_stage(group_keys)
        logger.info("[%s] %s: tuning + %d OOS seeds...", model_name, stage_name, n_runs)
        best_params = _tune_hyperparameters(features[columns], target, groups, cfg=cfg)

        seed_accumulators: list[tuple[np.ndarray, np.ndarray]] = Parallel(n_jobs=-1)(  # type: ignore[assignment]
            delayed(_collect_oos_predictions)(
                features[columns], target, groups, best_params, cfg, seed
            )
            for seed in range(n_runs)
        )
        total_sum = np.sum([s for s, _ in seed_accumulators], axis=0)
        total_count = np.sum([c for _, c in seed_accumulators], axis=0)
        y_oos_pred = (total_sum / np.maximum(total_count, 1)).clip(
            cfg.pe_min, cfg.pe_max
        )
        results[stage_name] = {"y_true": target, "y_oos_pred": y_oos_pred}

    return results
