DZ = {"mm": 1, "cm": 10, "dm": 100, "m": 1000}
DT = {
    "s": 1,
    "min": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
    "yr": 31536000,
    "30yr": 946080000,
    "50yr": 1576800000,
    "100yr": 3153600000,
}


def parse_dzdt_units(unit_str: str) -> tuple[str, str]:
    """
    Parse a unit string into its length and time components.

    Parameters
    ----------
    unit_str : str
        The unit string to parse (e.g., "mm/yr", "m/s").

    Returns
    -------
    tuple[str, str]
        A tuple containing the length and time units seperately (e.g., ("mm", "yr")).


    Raises
    ------
    ValueError
        If the unit string does not contain exactly one forward slash separator.

    """
    parts = unit_str.strip().split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid unit string: {unit_str}")
    dz, dt = parts
    return dz, dt


def calculate_dzdt_factor(dzdt_from: str, dzdt_to: str) -> float:
    """
    Calculate the conversion factor from one dz/dt unit to another.

    Parameters
    ----------
    dzdt_from : str
        The original dz/dt unit (e.g., "mm/yr").
    dzdt_to : str
        The target dz/dt unit (e.g., "m/yr").

    Returns
    -------
    float
        The conversion factor to convert from dzdt_from to dzdt_to.

    Raises
    ------
    ValueError
        If invalid units are provided (not given in the DZ or DT dictionaries defined
        above in this module).
    """
    dz_from, dt_from = parse_dzdt_units(dzdt_from)
    dz_to, dt_to = parse_dzdt_units(dzdt_to)

    if dz_from not in DZ or dt_from not in DT or dz_to not in DZ or dt_to not in DT:
        raise ValueError("Invalid units provided.")

    factor_dz = DZ[dz_from] / DZ[dz_to]
    factor_dt = DT[dt_from] / DT[dt_to]

    return factor_dz / factor_dt
