"""Shared I/O helpers for Snakemake rule scripts."""

import pickle
from pathlib import Path
from typing import Any

from swing3.config import DEFAULT_CFG, ExperimentConfig


def load_cfg(snakemake_config: dict) -> ExperimentConfig:
    """Build ExperimentConfig from Snakemake config dict, falling back to defaults."""
    cfg = DEFAULT_CFG
    fields = {
        "n_seeds",
        "n_optuna_trials",
        "n_folds",
        "early_stopping_rounds",
        "n_estimators",
        "temporal_cutoff",
        "hparam_val_size",
    }
    overrides = {k: snakemake_config[k] for k in fields if k in snakemake_config}
    if overrides:
        from dataclasses import replace

        cfg = replace(cfg, **overrides)
    return cfg


def save(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
