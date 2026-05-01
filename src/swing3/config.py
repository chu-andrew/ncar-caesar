from nc.remote import SWING3_MODELS

MODELS = list(SWING3_MODELS.keys())

PREDICTOR_GROUPS: dict[str, list[str]] = {
    "thermo": ["mcao", "sh", "qvsum", "q_700", "t_700", "ts"],
    "dynamics": ["wind_sfc", "iuq", "ivq", "omega_925", "omega_700"],
    "clouds": ["low_cloud"],
    "isotopes": ["dD_gradient", "dDp", "dexcessp"],
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

# Data source classification for each predictor variable.
# satellite     - retrievable from passive/active remote sensing
# reanalysis    - requires data assimilation or model-derived fields
# isotope_model - isotope-enabled model output; not directly observable
VARIABLE_DATA_SOURCES: dict[str, str] = {
    "low_cloud": "satellite",
    "ts": "satellite",
    "qvsum": "satellite",
    "q_700": "satellite",
    "t_700": "satellite",
    "mcao": "reanalysis",
    "sh": "reanalysis",
    "wind_sfc": "reanalysis",
    "iuq": "reanalysis",
    "ivq": "reanalysis",
    "omega_925": "reanalysis",
    "omega_700": "reanalysis",
    "dD_gradient": "isotope_model",
    "dDp": "isotope_model",
    "dexcessp": "isotope_model",
    "dDs": "isotope_model",
    "dexcesss": "isotope_model",
}
