"""Snakemake rule script: cross-model SHAP plots (all models combined)."""

import logging
from pathlib import Path

from swing3.rules._io import load
from swing3.plot_staged_shap import (
    plot_intermodel_heatmap,
    plot_mcao_dependence,
    plot_pe_scatter_by_stage,
    plot_r2_by_stage,
    plot_spatial_residuals,
    plot_stage_r2_bars,
)
from swing3.plot_feature_analysis import (
    compute_feature_stats,
    plot_direction_heatmap,
    plot_within_group_importance,
    print_direction_table,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

inp = snakemake.input  # type: ignore[name-defined]  # noqa: F821
models: list[str] = snakemake.params.models  # type: ignore[name-defined]  # noqa: F821

all_results = {m: load(p) for m, p in zip(models, inp.staged)}
oos_results = {m: load(p) for m, p in zip(models, inp.oos)}

plot_r2_by_stage(all_results)
plot_stage_r2_bars(all_results)
plot_intermodel_heatmap(all_results)
plot_mcao_dependence(all_results)
plot_spatial_residuals(all_results)
plot_pe_scatter_by_stage(all_results, oos_results)

feature_stats = compute_feature_stats(all_results)
plot_within_group_importance(feature_stats)
plot_direction_heatmap(feature_stats)
print_direction_table(feature_stats)

Path(snakemake.output[0]).touch()  # type: ignore[name-defined]  # noqa: F821
