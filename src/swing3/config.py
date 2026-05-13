from dataclasses import dataclass

from nc.remote import SWING3_MODELS

MODELS = list(SWING3_MODELS.keys())

# Isotope feature lists used across the pipeline.
# full:   all five isotope predictors -- Stage 4 and subgroup analysis
# no_ddp: full minus dDp -- circularity-test and forward selection
#         (dDp is co-determined with PE and excluded from attribution experiments)
ISOTOPE_EXTENDED: list[str] = ["dD_gradient", "dDp", "dexcessp", "dDs", "dexcesss"]
ISOTOPE_NO_DDP: list[str] = ["dD_gradient", "dexcessp", "dDs", "dexcesss"]

PREDICTOR_GROUPS: dict[str, list[str]] = {
    "thermo": ["mcao", "sh", "qvsum", "q_700", "t_700", "ts"],
    "dynamics": ["wind_sfc", "ivt", "omega_925", "omega_700"],
    "clouds": ["low_cloud"],
    "isotopes": ISOTOPE_EXTENDED,
}

STAGED_MODELS: list[tuple[str, list[str]]] = [
    ("Stage 1: Thermo", ["thermo"]),
    ("Stage 2: + Dynamics", ["thermo", "dynamics"]),
    ("Stage 3: + Clouds", ["thermo", "dynamics", "clouds"]),
    ("Stage 4: + Isotopes", ["thermo", "dynamics", "clouds", "isotopes"]),
]

# Groups to exclude entirely for specific models (e.g. CAM5 has no cloud data).
MODEL_EXCLUDED_GROUPS: dict[str, set[str]] = {
    "CAM5": {"clouds"},
}


def columns_for_stage(groups: list[str]) -> list[str]:
    """Return the flat list of feature column names for the given group keys."""
    return [col for g in groups for col in PREDICTOR_GROUPS[g]]


def stages_for_model(model_name: str) -> list[tuple[str, list[str]]]:
    """Return STAGED_MODELS filtered to exclude this model's excluded groups.

    A stage is dropped if it adds no new (non-excluded) groups relative to the
    previous included stage. E.g. CAM5 (excluded: clouds) gets:
      Stage 1, Stage 2, Stage 4  (Stage 3 is skipped because its only new group
      is clouds).
    """
    excluded = MODEL_EXCLUDED_GROUPS.get(model_name, set())
    result: list[tuple[str, list[str]]] = []
    prev_groups: set[str] = set()
    for stage_name, group_keys in STAGED_MODELS:
        filtered = [g for g in group_keys if g not in excluded]
        new_groups = set(filtered) - prev_groups
        if new_groups:
            result.append((stage_name, filtered))
            prev_groups = set(filtered)
    return result


GROUP_COLORS: dict[str, str] = {
    "thermo": "tab:red",
    "dynamics": "tab:blue",
    "clouds": "tab:green",
    "isotopes": "tab:purple",
}

GROUP_LABELS: dict[str, str] = {
    "thermo": "Thermodynamics",
    "dynamics": "Dynamics",
    "clouds": "Clouds",
    "isotopes": "Isotopes",
}


@dataclass(frozen=True)
class ExperimentConfig:
    n_seeds: int = 10
    n_folds: int = 5
    n_optuna_trials: int = 20
    early_stopping_rounds: int = 20
    n_estimators: int = 500
    temporal_cutoff: int = 2012
    # hparam_val_size: used only for the Optuna HPO inner CV (3 folds, 30% test each).
    # The early stopping split is separate and hardcoded at 0.2 in train_and_explain.
    hparam_val_size: float = 0.3
    pe_min: float = 0.0
    pe_max: float = 100.0


DEFAULT_CFG = ExperimentConfig()
