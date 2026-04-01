import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from joblib import Parallel, delayed
from scipy.stats import randint, uniform
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV

from nc.cache import MEMORY
from swing3.config import STAGED_MODELS, columns_for_stage
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
) -> dict:
    """Train XGBoost on one random split and compute SHAP values on the full dataset.

    Each split is 3-way (train / val / test): val is carved from train and used
    solely for early stopping. R^2 is evaluated on the held-out test set across 5
    outer folds. The model from the first fold is used for SHAP.
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

    model.fit(
        X.iloc[shap_tr_idx],
        y[shap_tr_idx],
        eval_set=[(X.iloc[shap_val_idx], y[shap_val_idx])],
        verbose=False,
    )
    shap_values = shap.TreeExplainer(model)(X)
    y_pred = model.predict(X).clip(0, 100)

    return {
        "shap_values": shap_values,
        "X_full": X,
        "y_full": y,
        "r2_train_mean": np.mean(r2_train_list),
        "r2_test_mean": np.mean(r2_test_list),
        "r2_train_std": np.std(r2_train_list),
        "r2_test_std": np.std(r2_test_list),
        "feature_names": columns,
        "residuals": y - y_pred,
        "best_params": best_params,
    }


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
    for stage_name, group_keys in STAGED_MODELS:
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
