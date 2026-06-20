"""Snakemake rule script: group Shapley sensitivity without dDp for one model."""

import logging

from swing3.group_shapley_attribution import run_group_shapley_attribution
from swing3.rules._io import load_cfg, save

logging.basicConfig(level=logging.INFO, format="%(message)s")

model_name: str = snakemake.wildcards.model  # type: ignore[name-defined]  # noqa: F821
cfg = load_cfg(snakemake.config)  # type: ignore[name-defined]  # noqa: F821

result = run_group_shapley_attribution(
    model_name,
    n_seeds=cfg.n_seeds,
    isotope_features=("dD_gradient", "dexcessp", "dDs", "dexcesss"),
)
save(result, snakemake.output[0])  # type: ignore[name-defined]  # noqa: F821
