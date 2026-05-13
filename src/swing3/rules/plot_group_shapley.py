"""Snakemake rule script: group Shapley attribution plots for all models combined."""

import logging
from pathlib import Path

from swing3.rules._io import load
from swing3.group_shapley_attribution import (
    plot_group_fraction_heatmap,
    plot_group_shapley_attribution,
    print_comparison_table,
    print_dominant_group_table,
    print_sensitivity_comparison,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

inp = snakemake.input  # type: ignore[name-defined]  # noqa: F821
models: list[str] = snakemake.params.models  # type: ignore[name-defined]  # noqa: F821

all_results = {m: load(p) for m, p in zip(models, inp.group_shapley)}
sensitivity_results = {m: load(p) for m, p in zip(models, inp.group_shapley_no_ddp)}

plot_group_shapley_attribution(all_results)
plot_group_fraction_heatmap(all_results)
print_comparison_table(all_results)
print_dominant_group_table(all_results)
print_sensitivity_comparison(all_results, sensitivity_results)

Path(snakemake.output[0]).touch()  # type: ignore[name-defined]  # noqa: F821
