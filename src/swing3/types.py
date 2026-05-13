"""Shared TypedDict result containers for all swing3 analysis modules."""

from typing import Any, TypedDict

import numpy as np
import pandas as pd
import shap


class RunResult(TypedDict):
    """Single-seed output from train_and_explain (no SHAP)."""

    r2_train_mean: float
    r2_test_mean: float
    r2_train_std: float
    r2_test_std: float
    feature_names: list[str]
    best_params: dict[str, Any]


class RunResultWithShap(RunResult):
    """Single-seed output from train_and_explain with SHAP values."""

    shap_values: shap.Explanation
    X_full: pd.DataFrame
    y_full: np.ndarray
    residuals: np.ndarray


class StagedResult(TypedDict):
    """Aggregated result from _aggregate_runs (SHAP variant)."""

    shap_values: shap.Explanation
    X_full: pd.DataFrame
    y_full: np.ndarray
    r2_train_mean: float
    r2_test_mean: float
    r2_train_std: float
    r2_test_std: float
    feature_names: list[str]
    residuals: np.ndarray
    n_runs: int


class ShapleyResult(TypedDict):
    """Output from run_group_shapley_attribution."""

    shapley: dict[str, float]
    shapley_std: dict[str, float]
    coalition_r2: dict[frozenset[str], float]
    stage_4_r2: float


class IsotopeSubgroupResult(TypedDict):
    """Output from run_isotope_subgroup_shapley."""

    shapley: dict[str, float]
    shapley_std: dict[str, float]
    coalition_r2: dict[frozenset[str], float]
    isotope_r2: float


class ForwardSelectionStep(TypedDict):
    k: int
    feature: str
    selected: list[str]
    all_candidate_r2: dict[str, tuple[float, float]]


class ForwardSelectionResult(TypedDict):
    """Output from run_stability_forward_selection."""

    selection_order: list[str]
    r2_mean_by_k: dict[int, float]
    r2_std_by_k: dict[int, float]
    steps: list[ForwardSelectionStep]


class TemporalRunResult(TypedDict):
    """Single-seed output from _train_one_seed in predict.py."""

    r2_train: float
    r2_test: float
    y_pred_test: np.ndarray
    y_true_test: np.ndarray
    years_test: np.ndarray


class OOSResult(TypedDict):
    """Per-stage out-of-sample prediction from run_staged_oos_predictions."""

    y_true: np.ndarray
    y_oos_pred: np.ndarray


class TemporalResult(TypedDict):
    """Aggregated output from _aggregate_temporal_runs."""

    r2_inner_mean: float
    r2_inner_std: float
    r2_test_mean: float
    r2_test_std: float
    y_pred_test: np.ndarray
    y_true_test: np.ndarray
    years_test: np.ndarray
    r2_per_seed: list[float]
    features_used: list[str]
    n_features: int
    n_runs: int
