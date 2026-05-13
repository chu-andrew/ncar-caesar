"""Snakemake rule script: isotope-only SHAP model for one model."""

import logging

from swing3.isotope_shap import run_isotope_only_analysis
from swing3.rules._io import load_cfg, save

logging.basicConfig(level=logging.INFO, format="%(message)s")

model_name: str = snakemake.wildcards.model  # type: ignore[name-defined]  # noqa: F821
cfg = load_cfg(snakemake.config)  # type: ignore[name-defined]  # noqa: F821

result = run_isotope_only_analysis(model_name, n_runs=cfg.n_seeds, cfg=cfg)
save(result, snakemake.output[0])  # type: ignore[name-defined]  # noqa: F821
