from dataclasses import dataclass


@dataclass(frozen=True)
class _Ds638001Vars:
    dataset: str = "638-001"
    time: str = "Time"
    latitude: str = "LATC"
    longitude: str = "LONC"
    altitude: str = "GGALT"
    surface_temp: str = "RSTB"


@dataclass(frozen=True)
class _Ds638021Vars:
    dataset: str = "638-021"
    time: str = "time"
    altitude: str = "Alt"
    bin_heights: str = "H"
    temperature: str = "T"


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


DS_638_001 = _Ds638001Vars()
DS_638_021 = _Ds638021Vars()
DS_638_038 = _Ds638038Vars()
DS_638_052 = _Ds638052Vars()
MICROPHYSICS = _MicrophysicsVars()
