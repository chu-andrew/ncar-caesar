"""Snakemake rule script: isotope-only beeswarm plots for all models."""

import logging
import os
from pathlib import Path

from nc.loader import PROJECT_ROOT
from swing3.rules._io import load
from swing3.isotope_shap import plot_isotope_beeswarms

logging.basicConfig(level=logging.INFO, format="%(message)s")

inp = snakemake.input  # type: ignore[name-defined]  # noqa: F821
models: list[str] = snakemake.params.models  # type: ignore[name-defined]  # noqa: F821

os.makedirs(
    os.path.join(PROJECT_ROOT, "output/swing3/plots/shap/isotope"), exist_ok=True
)

all_isotope = {m: load(p) for m, p in zip(models, inp.isotope_only)}

plot_isotope_beeswarms(all_isotope)

Path(snakemake.output[0]).touch()  # type: ignore[name-defined]  # noqa: F821
