"""Snakemake rule script: per-model SHAP plots (beeswarms + print diagnostics)."""

import logging
from pathlib import Path

from swing3.rules._io import load
from swing3.plot_staged_shap import plot_beeswarms
from swing3.plot_isotope_sensitivity import print_forward_comparison_table

logging.basicConfig(level=logging.INFO, format="%(message)s")

inp = snakemake.input  # type: ignore[name-defined]  # noqa: F821
model_name: str = snakemake.wildcards.model  # type: ignore[name-defined]  # noqa: F821

staged_result = load(inp.staged)
oos_result = load(inp.oos)
forward_result = load(inp.forward_model)

all_results = {model_name: staged_result}
oos_results = {model_name: oos_result}
forward_results = {model_name: forward_result}

plot_beeswarms(all_results)
print_forward_comparison_table(all_results, forward_results)

Path(snakemake.output[0]).touch()  # type: ignore[name-defined]  # noqa: F821
