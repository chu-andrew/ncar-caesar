"""Snakemake rule script: temporal prediction plots for all models combined."""

import logging
from pathlib import Path

from swing3.rules._io import load
from swing3.plot_predict import (
    plot_pred_vs_actual,
    plot_residuals_by_year,
    plot_spatial_pe_comparison,
    print_summary_table,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

inp = snakemake.input  # type: ignore[name-defined]  # noqa: F821
models: list[str] = snakemake.params.models  # type: ignore[name-defined]  # noqa: F821

all_results = {m: load(p) for m, p in zip(models, inp.predictions)}

plot_pred_vs_actual(all_results)
plot_residuals_by_year(all_results)
plot_spatial_pe_comparison(all_results)
print_summary_table(all_results)

Path(snakemake.output[0]).touch()  # type: ignore[name-defined]  # noqa: F821
