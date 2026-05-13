"""Snakemake rule script: run staged SHAP analysis for one model."""

import logging

from swing3.rules._io import load_cfg, save
from swing3.shap_analysis import run_staged_analysis

logging.basicConfig(level=logging.INFO, format="%(message)s")

model_name: str = snakemake.wildcards.model  # type: ignore[name-defined]  # noqa: F821
cfg = load_cfg(snakemake.config)  # type: ignore[name-defined]  # noqa: F821

results = run_staged_analysis(model_name, n_runs=cfg.n_seeds, cfg=cfg)
save(results, snakemake.output[0])  # type: ignore[name-defined]  # noqa: F821
