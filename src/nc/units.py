import numpy as np

# temperature
ZERO_CELSIUS_IN_KELVIN = 273.15

# length
M_PER_KM = 1000.0
M_PER_UM = 1e-6
UM_PER_M = 1e6

# time
S_PER_HR = 3600.0
NS_PER_S = 1e9

# mass
KG_PER_G = 1e-3


def celsius_to_kelvin(t: float) -> float:
    return t + ZERO_CELSIUS_IN_KELVIN


def m_to_km(x: np.ndarray) -> np.ndarray:
    return x / M_PER_KM


def um_to_m(x: np.ndarray) -> np.ndarray:
    return x * M_PER_UM
