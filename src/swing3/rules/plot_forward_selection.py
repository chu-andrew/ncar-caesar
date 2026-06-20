"""Snakemake rule script: forward selection plots for all models combined."""

import logging
from pathlib import Path

from swing3.rules._io import load
from swing3.forward_selection import (
    plot_r2_vs_k,
    plot_selection_frequency,
    print_pareto_table,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

inp = snakemake.input  # type: ignore[name-defined]  # noqa: F821
models: list[str] = snakemake.params.models  # type: ignore[name-defined]  # noqa: F821

selection_results = {m: load(p) for m, p in zip(models, inp.fwd_sel)}

plot_r2_vs_k(selection_results)
plot_selection_frequency(selection_results)
print_pareto_table(selection_results)

Path(snakemake.output[0]).touch()  # type: ignore[name-defined]  # noqa: F821
