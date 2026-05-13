"""Snakemake rule script: temporal holdout prediction for one model."""

import logging

from swing3.predict import run_temporal_analysis
from swing3.rules._io import load_cfg, save

logging.basicConfig(level=logging.INFO, format="%(message)s")

model_name: str = snakemake.wildcards.model  # type: ignore[name-defined]  # noqa: F821
cfg = load_cfg(snakemake.config)  # type: ignore[name-defined]  # noqa: F821

result = run_temporal_analysis(
    model_name,
    n_runs=cfg.n_seeds,
    train_cutoff=cfg.temporal_cutoff,
    cfg=cfg,
)
save(result, snakemake.output[0])  # type: ignore[name-defined]  # noqa: F821
