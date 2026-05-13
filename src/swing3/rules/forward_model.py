"""Snakemake rule script: Stage 4 with dDp excluded (forward model sensitivity)."""

import logging

from swing3.rules._io import load_cfg, save
from swing3.shap_analysis import run_forward_model

logging.basicConfig(level=logging.INFO, format="%(message)s")

model_name: str = snakemake.wildcards.model  # type: ignore[name-defined]  # noqa: F821
cfg = load_cfg(snakemake.config)  # type: ignore[name-defined]  # noqa: F821

result = run_forward_model(model_name, n_runs=cfg.n_seeds, cfg=cfg)
save(result, snakemake.output[0])  # type: ignore[name-defined]  # noqa: F821
