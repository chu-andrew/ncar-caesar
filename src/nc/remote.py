import os

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
RDF_REMOTE = Path(os.getenv("RDF_REMOTE"))
RDF_DATA = RDF_REMOTE / "DATA"

WISONET_DATA = RDF_DATA / "wisonet"
SWING3_DATA = WISONET_DATA / "SWING3"
ERA5_DATA = SWING3_DATA / "ERA5"

ERA5_SST = ERA5_DATA / "SST" / "sst_all.nc"

SWING3_MODELS = {
    "CAM5": SWING3_DATA / "Total.CAM5_Monthly.nc",
    "CAM6": SWING3_DATA / "Total.CAM6_Monthly.nc",
    "ECHAM": SWING3_DATA / "Total.ECHAM_Monthly.nc",
    "GISS": SWING3_DATA / "Total.GISS_Monthly.nc",
    "GSM": SWING3_DATA / "Total.GSM_Monthly.nc",
    "LMDZ": SWING3_DATA
    / "Total.LMDZ_Monthly.nc",  # uses "time_counter" instead of "time"
    "MIROC": SWING3_DATA / "Total.MIROC_Monthly.nc",
    # "NICAM": SWING3_DATA / "Total.NICAM_Monthly.nc", # omitted
}
