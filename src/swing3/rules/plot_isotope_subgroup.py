"""Snakemake rule script: isotope sub-group Shapley plots for all models combined."""

import logging
from pathlib import Path

from swing3.rules._io import load
from swing3.isotope_subgroup import (
    plot_isotope_subgroup_attribution,
    print_subgroup_table,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

inp = snakemake.input  # type: ignore[name-defined]  # noqa: F821
models: list[str] = snakemake.params.models  # type: ignore[name-defined]  # noqa: F821

all_results = {m: load(p) for m, p in zip(models, inp.results)}

print_subgroup_table(all_results)
plot_isotope_subgroup_attribution(all_results)

Path(snakemake.output[0]).touch()  # type: ignore[name-defined]  # noqa: F821
