"""One-off script: export variable CSVs for each model's cloud folder."""

import os

from nc.remote import CLOUD_DIR, SWING3_MODELS
from nc.loader import list_dir_files, PROJECT_ROOT
from nc.variables import export_variable_groups

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/remote/swing3/variables/cloud")
MODELS = list(SWING3_MODELS.keys())


ECHAM6_FILES = [
    "NUDGING_ERA5_T63L47_v1.2_echam6_197901-202412.aclcac.monmean.nc",
    "NUDGING_ERA5_T63L47_v1.2_echam6_197901-202412.hih_cld.monmean.nc 2",
    "NUDGING_ERA5_T63L47_v1.2_echam6_197901-202412.low_cld.monmean.nc 2",
    "NUDGING_ERA5_T63L47_v1.2_echam6_197901-202412.mid_cld.monmean.nc 2",
]


for model in MODELS:
    if model == "ECHAM":
        model = "ECHAM6"

    model_dir = CLOUD_DIR / model
    if not model_dir.is_dir():
        print(f"[skip] {model}: directory not found ({model_dir})")
        continue

    if model == "ECHAM6":
        files = [model_dir / f for f in ECHAM6_FILES]
    else:
        files = list_dir_files(model_dir)

    if not files:
        print(f"[skip] {model}: no NetCDF files found")
        continue

    output_dir = os.path.join(OUTPUT_DIR, model)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== {model} ===")
    export_variable_groups(files, output_dir)
