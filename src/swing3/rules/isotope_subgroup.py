"""Snakemake rule script: isotope sub-group Shapley for one model."""

import logging

from swing3.isotope_subgroup import run_isotope_subgroup_shapley
from swing3.rules._io import load_cfg, save

logging.basicConfig(level=logging.INFO, format="%(message)s")

model_name: str = snakemake.wildcards.model  # type: ignore[name-defined]  # noqa: F821
cfg = load_cfg(snakemake.config)  # type: ignore[name-defined]  # noqa: F821

result = run_isotope_subgroup_shapley(model_name, n_seeds=cfg.n_seeds)
save(result, snakemake.output[0])  # type: ignore[name-defined]  # noqa: F821
