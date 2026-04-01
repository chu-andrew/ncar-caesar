from dataclasses import dataclass

from nc.remote import SWING3_DATA


@dataclass(frozen=True)
class _Ds638001Vars:
    dataset: str = "638-001"
    time: str = "Time"
    latitude: str = "LATC"
    longitude: str = "LONC"
    altitude: str = "GGALT"
    surface_temp: str = "RSTB"
    vmr_vxl: str = "VMR_VXL"
    theta: str = "THETA"
    pressure: str = "PSXC"


@dataclass(frozen=True)
class _Ds638021Vars:
    dataset: str = "638-021"
    time: str = "time"
    altitude: str = "Alt"
    bin_heights: str = "H"
    temperature: str = "T"
    wvmr: str = "WVMR"
    fill_value: float = 9999.0


@dataclass(frozen=True)
class _Ds638038Vars:
    dataset: str = "638-038"
    time: str = "time"
    altitude: str = "alt"
    lwp: str = "LWP"
    wvp: str = "WVP"


@dataclass(frozen=True)
class _Ds638052Vars:
    dataset: str = "638-052"
    time: str = "time"
    cloud_base: str = "cloudbase_WCL"


@dataclass(frozen=True)
class _MicrophysicsVars:
    time: str = "time"
    cloud_phase: str = "cloud_phase"
    concentration: str = "concentration"
    bin_edges: str = "bin_edges"


@dataclass(frozen=True)
class _Swing3SSTVars:
    time: str = "valid_time"
    sst: str = "sst"
    lat: str = "lat"
    lon: str = "lon"


@dataclass(frozen=True)
class _Swing3Vars:
    time: str = "time"
    temperature: str = "t"
    pressure: str = "p"
    precip_efficiency: str = "pref"
    lat: str = "lat"
    lon: str = "lon"
    specific_humidity: str = "q"
    surface_specific_humidity: str = "sh"
    precipitable_water: str = "qvsum"
    surface_temperature: str = "ts"
    surface_pressure: str = "ps"
    u_wind: str = "u"
    v_wind: str = "v"
    moisture_flux_u: str = "iuq"
    moisture_flux_v: str = "ivq"
    precipitation: str = "pr"
    evaporation: str = "ev"
    dD_vapor: str = "dD"
    dD_precip: str = "dDp"
    dexcess_precip: str = "dexcessp"


@dataclass(frozen=True)
class _Swing3LMDZVars:
    time: str = "time_counter"  # LMDZ uses a non-standard time dimension name


DS_638_001 = _Ds638001Vars()
DS_638_021 = _Ds638021Vars()
DS_638_038 = _Ds638038Vars()
DS_638_052 = _Ds638052Vars()
MICROPHYSICS = _MicrophysicsVars()
SWING3_SST = _Swing3SSTVars()
SWING3 = _Swing3Vars()
SWING3_LMDZ = _Swing3LMDZVars()
