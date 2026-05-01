"""Temporal holdout prediction model for precipitation efficiency (PE).

Trains on GCM data (1979-2012) and evaluates on a held-out period (2013-2021).
Uses features selected by forward selection at a 95% R2 threshold, split into
two scenarios:
  - observable: satellite + reanalysis features only (usable for 2024+ prediction)
  - all: all selected features including isotopes (GCM data range only)
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit

from nc.cache import MEMORY
from nc.loader import PROJECT_ROOT
from swing3.config import MODELS, VARIABLE_DATA_SOURCES
from swing3.features import load_predict_features
from swing3.forward_selection import run_stability_forward_selection
from swing3.shap_analysis import _tune_hyperparameters

PREDICT_PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/remote/swing3/plots/predict")

_OBSERVABLE_SOURCES = {"satellite", "reanalysis"}

_FSEL_CACHE_DIRS = [
    os.path.join(PROJECT_ROOT, "cache", d, "run_stability_forward_selection")
    for d in os.listdir(os.path.join(PROJECT_ROOT, "cache"))
    if "forward_selection" in d
] if os.path.exists(os.path.join(PROJECT_ROOT, "cache")) else []


def _load_forward_selection_result(model_name: str, n_seeds: int) -> dict | None:
    """Search all known cache locations for a matching forward selection result."""
    for cache_dir in _FSEL_CACHE_DIRS:
        if not os.path.isdir(cache_dir):
            continue
        for entry in os.listdir(cache_dir):
            meta_path = os.path.join(cache_dir, entry, "metadata.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            args = meta.get("input_args", {})
            if (args.get("model_name") == f"'{model_name}'"
                    and args.get("n_seeds") == str(n_seeds)):
                with open(os.path.join(cache_dir, entry, "output.pkl"), "rb") as f:
                    return pickle.load(f)
    return None


def get_selected_features(
    model_name: str,
    r2_threshold: float = 0.95,
    n_seeds: int = 5,
) -> tuple[list[str], list[str]]:
    """Return (observable_cols, all_cols) from forward selection at r2_threshold.

    Finds the smallest k such that r2_mean_by_k[k] >= r2_threshold * max_r2,
    then returns the top-k selected features split by data source.

    observable_cols: non-isotope features from the selected set
    all_cols:        all selected features (including isotopes if present in top-k)
    """
    result = _load_forward_selection_result(model_name, n_seeds)
    if result is None:
        result = run_stability_forward_selection(model_name, n_seeds=n_seeds)
    selection_order = result["selection_order"]
    r2_by_k = result["r2_mean_by_k"]

    max_r2 = r2_by_k[max(r2_by_k)]
    cutoff_r2 = r2_threshold * max_r2

    k_cutoff = max(r2_by_k)
    for k in sorted(r2_by_k):
        if r2_by_k[k] >= cutoff_r2:
            k_cutoff = k
            break

    all_cols = selection_order[:k_cutoff]
    observable_cols = [
        c for c in all_cols
        if VARIABLE_DATA_SOURCES.get(c, "reanalysis") in _OBSERVABLE_SOURCES
    ]
    return observable_cols, all_cols


def temporal_split(
    years: np.ndarray,
    train_cutoff: int = 2012,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean (train_mask, test_mask) based on calendar year."""
    train_mask = years <= train_cutoff
    test_mask = years > train_cutoff
    return train_mask, test_mask


def _train_one_seed(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    years_test: np.ndarray,
    best_params: dict,
    seed: int = 0,
) -> dict:
    """Train on training data with inner val fold for early stopping; evaluate on test."""
    inner_gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, val_idx = next(inner_gss.split(X_train, y_train, groups_train))

    model = xgb.XGBRegressor(
        n_estimators=500,
        tree_method="hist",
        random_state=seed,
        early_stopping_rounds=20,
        **best_params,
    )
    model.fit(
        X_train.iloc[tr_idx],
        y_train[tr_idx],
        eval_set=[(X_train.iloc[val_idx], y_train[val_idx])],
        verbose=False,
    )

    y_pred_test = model.predict(X_test).clip(0, 100)

    return {
        "r2_train": model.score(X_train.iloc[tr_idx], y_train[tr_idx]),
        "r2_test": model.score(X_test, y_test),
        "y_pred_test": y_pred_test,
        "y_true_test": y_test,
        "years_test": years_test,
    }


def _aggregate_temporal_runs(run_results: list[dict], features_used: list[str]) -> dict:
    """Average N seed results into ensemble metrics and mean predictions.

    r2_inner_std and r2_test_std reflect model variance across seeds (each seed uses a
    different early-stopping val fold), not test-set sampling variance.
    """
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
) -> dict[str, dict]:
    """Temporal holdout prediction for one climate model.

    Returns dict with keys "observable" and "all", each containing aggregated
    metrics and predictions for the test period (train_cutoff+1 to 2023).
    """
    features, target, groups, years = load_predict_features(model_name)

    if years.min() != 1979:
        raise ValueError(f"[{model_name}] Unexpected start year: {years.min()}, expected 1979")
    if years.max() <= train_cutoff:
        raise ValueError(f"[{model_name}] No test data: all years <= {train_cutoff}")

    train_mask, test_mask = temporal_split(years, train_cutoff)
    n_train = train_mask.sum()
    n_test = test_mask.sum()
    n_test_years = int(years[test_mask].max() - train_cutoff)
    n_independent = n_test_years * 4
    print(f"[{model_name}] {len(features):,} samples -- train={n_train:,} (<={train_cutoff}), "
          f"test={n_test:,} (>{train_cutoff}, {n_independent} independent time points)")

    observable_cols, all_cols = get_selected_features(model_name, r2_threshold)
    print(f"[{model_name}] Forward selection @ {r2_threshold:.0%}: "
          f"{len(observable_cols)} observable, {len(all_cols)} total features")
    print(f"[{model_name}]   observable: {observable_cols}")
    print(f"[{model_name}]   all:        {all_cols}")

    results = {}
    for scenario, columns in [("observable", observable_cols), ("all", all_cols)]:
        if not columns:
            print(f"[{model_name}] Skipping '{scenario}': no features selected")
            continue

        X = features[columns]
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = target[train_mask]
        y_test = target[test_mask]
        groups_train = groups[train_mask]
        years_test = years[test_mask]

        print(f"[{model_name}] {scenario}: tuning hyperparameters on training data...")
        best_params = _tune_hyperparameters(X_train, y_train, groups_train)

        print(f"[{model_name}] {scenario}: running {n_runs} seeds...")
        run_results = Parallel(n_jobs=-1)(
            delayed(_train_one_seed)(
                X_train, y_train, groups_train,
                X_test, y_test, years_test,
                best_params, seed=s,
            )
            for s in range(n_runs)
        )
        results[scenario] = _aggregate_temporal_runs(run_results, columns)
        r = results[scenario]
        print(f"[{model_name}] {scenario}: R2 inner fold={r['r2_inner_mean']:.3f} (+-{r['r2_inner_std']:.3f}), "
              f"test={r['r2_test_mean']:.3f} (+-{r['r2_test_std']:.3f})")

    return results


def main() -> None:
    os.makedirs(PREDICT_PLOTS_DIR, exist_ok=True)

    all_results = {}
    for model_name in MODELS:
        print(f"\n=== {model_name} ===")
        all_results[model_name] = run_temporal_analysis(model_name)

    print("\n=== Summary ===")
    print(f"{'Model':<10}  {'Obs feats':>9}  {'Obs R2':>8}  {'All feats':>9}  {'All R2':>8}  {'dR2':>6}")
    print("-" * 60)
    for model_name, res in all_results.items():
        obs = res.get("observable", {})
        all_ = res.get("all", {})
        n_obs = obs.get("n_features", 0)
        n_all = all_.get("n_features", 0)
        r2_obs = obs.get("r2_test_mean", float("nan"))
        r2_all = all_.get("r2_test_mean", float("nan"))
        delta = r2_all - r2_obs if not (np.isnan(r2_obs) or np.isnan(r2_all)) else float("nan")
        print(f"{model_name:<10}  {n_obs:>9}  {r2_obs:>8.3f}  {n_all:>9}  {r2_all:>8.3f}  {delta:>+6.3f}")


if __name__ == "__main__":
    main()
