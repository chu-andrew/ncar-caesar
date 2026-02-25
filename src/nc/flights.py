"""Centralized flight-to-date mapping and MARLi filename resolution."""

FLIGHTS = {
    "RF01": "2024-02-28",
    "RF02": "2024-02-29",
    "RF03": "2024-03-02",
    "RF04": "2024-03-05",
    "RF05": "2024-03-11",
    "RF06": "2024-03-12",
    "RF07": "2024-03-16",
    "RF09": "2024-04-02",
    "RF10": "2024-04-03",
}

MARLI_FILES = {
    "RF01": ["CAESAR-MARLi_C130_20240228_R0.cdf"],
    "RF02": ["CAESAR-MARLi_C130_20240229_R0.cdf"],
    "RF03": ["CAESAR-MARLi_C130_20240302_R0.cdf"],
    "RF04": ["CAESAR-MARLi_C130_20240305_R0.cdf"],
    "RF05": ["CAESAR-MARLi_C130_20240311_R0.cdf"],
    "RF06": ["CAESAR-MARLi_C130_20240312_R0.cdf"],
    "RF07": [
        "CAESAR-MARLi_C130_20240316_part1_R0.cdf",
        "CAESAR-MARLi_C130_20240316_part2_R0.cdf",
    ],
    "RF09": ["CAESAR-MARLi_C130_20240402_R0.cdf"],
    "RF10": ["CAESAR-MARLi_C130_20240403_R0.cdf"],
}
