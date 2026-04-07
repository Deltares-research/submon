DZ = {"mm": 1, "cm": 10, "dm": 100, "m": 1000}
DT = {
    "s": 1,
    "min": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
    "yr": 31536000,
    "30yr": 946080000,
}


def parse_dzdt_units(unit_str: str) -> tuple[str, str]:
    """Parse a unit string into its value and unit components.

    Args:
        unit_str: The unit string to parse (e.g., "10 mm", "5 min").

    Returns:
        A tuple containing the value and the unit (e.g., ("10", "mm")).
    """
    parts = unit_str.strip().split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid unit string: {unit_str}")
    dz, dt = parts
    return dz, dt


def calculate_dzdt_factor(dzdt_from: str, dzdt_to: str) -> float:
    """Calculate the conversion factor from one dz/dt unit to another.

    Args:
        dzdt_from: The original dz/dt unit (e.g., "mm/yr").
        dzdt_to: The target dz/dt unit (e.g., "m/yr").

    Returns:
        The conversion factor to convert from dzdt_from to dzdt_to.
    """
    dz_from, dt_from = parse_dzdt_units(dzdt_from)
    dz_to, dt_to = parse_dzdt_units(dzdt_to)

    if dz_from not in DZ or dt_from not in DT or dz_to not in DZ or dt_to not in DT:
        raise ValueError("Invalid units provided.")

    factor_dz = DZ[dz_from] / DZ[dz_to]
    factor_dt = DT[dt_from] / DT[dt_to]

    return factor_dz / factor_dt
