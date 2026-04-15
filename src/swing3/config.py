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


def columns_for_stage(groups: list[str]) -> list[str]:
    """Return the flat list of feature column names for the given group keys."""
    return [col for g in groups for col in PREDICTOR_GROUPS[g]]


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
