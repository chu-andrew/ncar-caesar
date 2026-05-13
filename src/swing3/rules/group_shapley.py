"""Snakemake rule script: group Shapley attribution for one model."""

import logging

from swing3.config import MODEL_EXCLUDED_GROUPS
from swing3.group_shapley_attribution import run_group_shapley_attribution
from swing3.rules._io import load_cfg, save

logging.basicConfig(level=logging.INFO, format="%(message)s")

model_name: str = snakemake.wildcards.model  # type: ignore[name-defined]  # noqa: F821
cfg = load_cfg(snakemake.config)  # type: ignore[name-defined]  # noqa: F821

excluded = tuple(sorted(MODEL_EXCLUDED_GROUPS.get(model_name, set())))
result = run_group_shapley_attribution(
    model_name, n_seeds=cfg.n_seeds, excluded_groups=excluded or None
)
save(result, snakemake.output[0])  # type: ignore[name-defined]  # noqa: F821
