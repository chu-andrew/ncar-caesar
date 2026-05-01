import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from joblib import Parallel, delayed
from scipy.stats import randint, uniform
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV

from nc.cache import MEMORY
from swing3.config import columns_for_stage, stages_for_model
from swing3.features import load_shap_features

_PARAM_DIST = {
    "max_depth": randint(3, 6),
    "learning_rate": uniform(0.01, 0.15),
    "subsample": uniform(0.5, 0.4),  # row sampling (0.5–0.9)
    "colsample_bytree": uniform(0.5, 0.5),
    "min_child_weight": randint(1, 15),
    "gamma": uniform(0, 5),  # minimum loss reduction to split
    "reg_alpha": uniform(0, 5),  # L1 regularization
    "reg_lambda": uniform(1, 4),  # L2 regularization; starts at XGBoost default of 1
}


def _tune_hyperparameters(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    random_state: int = 0,
    n_jobs: int = -1,
) -> dict:
    """Run RandomizedSearchCV on one GroupShuffleSplit train fold and return best params."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
    train_idx, _ = next(gss.split(X, y, groups))

    base_model = xgb.XGBRegressor(
        n_estimators=500, tree_method="hist", random_state=random_state
    )
    # inner cross-validation must also use groups to prevent spatial leakage during tuning
    inner_cv = GroupShuffleSplit(n_splits=3, test_size=0.3, random_state=random_state)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=_PARAM_DIST,
        n_iter=50,
        cv=inner_cv,
        scoring="r2",
        n_jobs=n_jobs,
        random_state=random_state,
    )
    search.fit(X.iloc[train_idx], y[train_idx], groups=groups[train_idx])
    return search.best_params_


def train_and_explain(
    features: pd.DataFrame,
    target: np.ndarray,
    columns: list[str],
    groups: np.ndarray,
    best_params: dict,
    random_state: int = 0,
    compute_shap: bool = True,
) -> dict:
    """Train XGBoost across 5 outer CV folds and optionally compute SHAP values.

    Each split is 3-way (train / val / test): val is carved from train and used
    solely for early stopping. R^2 is evaluated on the held-out test set across 5
    outer folds. The model from the first fold is used for SHAP (when compute_shap=True).

    When compute_shap=False, only R^2 statistics are returned (no shap_values,
    X_full, y_full, or residuals keys). Used by group_shapley_attribution.py to evaluate
    coalition R^2 without the overhead of SHAP computation.
    """
    X = features[columns]
    y = target

    model = xgb.XGBRegressor(
        n_estimators=500,  # upper bound; early stopping determines actual count
        tree_method="hist",
        random_state=random_state,
        early_stopping_rounds=20,
        **best_params,
    )

    outer_gss = GroupShuffleSplit(n_splits=5, test_size=0.3, random_state=random_state)
    inner_gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)

    r2_train_list, r2_test_list = [], []
    shap_tr_idx = shap_val_idx = None

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
        r2_train_list.append(m_cv.score(X.iloc[tr_idx], y[tr_idx]))
        r2_test_list.append(m_cv.score(X.iloc[test_idx], y[test_idx]))

    result = {
        "r2_train_mean": np.mean(r2_train_list),
        "r2_test_mean": np.mean(r2_test_list),
        "r2_train_std": np.std(r2_train_list),
        "r2_test_std": np.std(r2_test_list),
        "feature_names": columns,
        "best_params": best_params,
    }

    if compute_shap:
        model.fit(
            X.iloc[shap_tr_idx],
            y[shap_tr_idx],
            eval_set=[(X.iloc[shap_val_idx], y[shap_val_idx])],
            verbose=False,
        )
        shap_values = shap.TreeExplainer(model)(X)
        y_pred = model.predict(X).clip(0, 100)
        result.update(
            {
                "shap_values": shap_values,
                "X_full": X,
                "y_full": y,
                "residuals": y - y_pred,
            }
        )

    return result


def _aggregate_runs(run_results: list[dict]) -> dict:
    """
    Average N train_and_explain results into one ensemble result.
    """
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
        "r2_train_mean": np.mean([r["r2_train_mean"] for r in run_results]),
        "r2_test_mean": np.mean([r["r2_test_mean"] for r in run_results]),
        "r2_train_std": np.std([r["r2_train_mean"] for r in run_results]),
        "r2_test_std": np.std([r["r2_test_mean"] for r in run_results]),
        "feature_names": first["feature_names"],
        "residuals": residuals,
        "n_runs": len(run_results),
    }


@MEMORY.cache
def run_staged_analysis(model_name: str, n_runs: int = 25) -> dict[str, dict]:
    """Run all 4 staged SHAP analyses for one climate model, averaged over n_runs seeds."""
    print(f"Loading features for {model_name}...")
    features, target, groups = load_shap_features(model_name)
    print(f"\t{model_name}: {len(features):,} samples")

    results = {}
    for stage_name, group_keys in stages_for_model(model_name):
        columns = columns_for_stage(group_keys)
        print(f"\t{stage_name} ({len(columns)} predictors, {n_runs} runs)...")

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
        results[stage_name] = _aggregate_runs(run_results)
        r = results[stage_name]
        print(
            f"\t\tR² train={r['r2_train_mean']:.3f} (±{r['r2_train_std']:.3f}), "
            f"test={r['r2_test_mean']:.3f} (±{r['r2_test_std']:.3f})"
        )

    return results


@MEMORY.cache
def run_forward_model(model_name: str, n_runs: int = 25) -> dict:
    """Stage 4 model with dDp excluded; tests whether dDp drives the attribution."""
    columns = [c for c in columns_for_stage(["thermo", "dynamics", "clouds", "isotopes"])
               if c != "dDp"]
    return _run_isotope_variant(model_name, columns, "dDp excluded", n_runs)


def _run_isotope_variant(
    model_name: str, columns: list[str], label: str, n_runs: int
) -> dict:
    """Shared implementation for surface isotope comparison runs."""
    print(f"Loading features for {model_name}...")
    features, target, groups = load_shap_features(model_name)
    print(f"\t{model_name}: {len(features):,} samples, {len(columns)} features ({label})")

    print("\t\tTuning hyperparameters...")
    best_params = _tune_hyperparameters(features[columns], target, groups)

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
        f"\t\tR² train={result['r2_train_mean']:.3f} (±{result['r2_train_std']:.3f}), "
        f"test={result['r2_test_mean']:.3f} (±{result['r2_test_std']:.3f})"
    )
    return result


def _final_stage_groups(model_name: str) -> list[str]:
    """Return the group keys for this model's final stage."""
    return stages_for_model(model_name)[-1][1]


def _collect_oos_predictions(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    best_params: dict,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run 5-fold CV and return (pred_sum, pred_count) for all points.

    Returns raw accumulators rather than a divided array so the caller can
    aggregate across seeds before dividing, avoiding bias from points that
    happen to fall outside all test folds in a given seed.
    """
    model = xgb.XGBRegressor(
        n_estimators=500,
        tree_method="hist",
        random_state=random_state,
        early_stopping_rounds=20,
        **best_params,
    )
    outer_gss = GroupShuffleSplit(n_splits=5, test_size=0.3, random_state=random_state)
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
def run_staged_oos_predictions(model_name: str, n_runs: int = 25) -> dict[str, dict]:
    """OOS predictions for each stage, averaged over n_runs seeds.

    Separate from run_staged_analysis so the SHAP cache is not disturbed.
    Returns: stage_name → {"y_true": ..., "y_oos_pred": ...}
    """
    features, target, groups = load_shap_features(model_name)

    results = {}
    for stage_name, group_keys in stages_for_model(model_name):
        columns = columns_for_stage(group_keys)
        print(f"\t[{model_name}] {stage_name}: tuning + {n_runs} OOS seeds...")
        best_params = _tune_hyperparameters(features[columns], target, groups)

        seed_accumulators = Parallel(n_jobs=-1)(
            delayed(_collect_oos_predictions)(
                features[columns], target, groups, best_params, seed
            )
            for seed in range(n_runs)
        )
        total_sum = np.sum([s for s, _ in seed_accumulators], axis=0)
        total_count = np.sum([c for _, c in seed_accumulators], axis=0)
        y_oos_pred = (total_sum / np.maximum(total_count, 1)).clip(0, 100)
        results[stage_name] = {
            "y_true": target,
            "y_oos_pred": y_oos_pred,
        }

    return results


@MEMORY.cache
def run_surface_isotopes_added(model_name: str, n_runs: int = 25) -> dict:
    """Final stage + dDs + dexcesss added to isotope group (Case A).

    Tests whether surface vapor isotopes add predictive skill on top of the
    existing isotope features (dD_gradient, dDp, dexcessp). Uses model-specific
    final stage groups (e.g. CAM5 excludes clouds).
    """
    base_cols = columns_for_stage(_final_stage_groups(model_name))
    columns = base_cols + ["dDs", "dexcesss"]
    return _run_isotope_variant(model_name, columns, "+dDs+dexcesss", n_runs)


@MEMORY.cache
def run_surface_isotopes_replace(model_name: str, n_runs: int = 25) -> dict:
    """Final stage with dDp replaced by dDs + dexcesss (Case B).

    Tests whether forward-only surface vapor isotopes (dDs, dexcesss) can
    substitute for the potentially circular dDp without loss of predictive skill.
    Uses model-specific final stage groups (e.g. CAM5 excludes clouds).
    """
    base_cols = columns_for_stage(_final_stage_groups(model_name))
    columns = [c for c in base_cols if c != "dDp"] + ["dDs", "dexcesss"]
    return _run_isotope_variant(model_name, columns, "dDp→dDs+dexcesss", n_runs)
