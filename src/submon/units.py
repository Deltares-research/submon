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


def dzdt_to_dz_over_period(
    value: float,
    uncertainty: float,
    current_unit: str,
    desired_length_unit: str,
    years: float,
) -> tuple[float, float]:
    """
    Convert a dz/dt value (e.g. mm/yr) to a length over a given period
    (e.g. cm over X years), including uncertainty.

    Parameters
    ----------
    value : float
        Mean dz/dt value.
    uncertainty : float
        Uncertainty in dz/dt.
    current_unit : str
        Unit of value and uncertainty, e.g. "mm/yr".
    desired_length_unit : str
        Desired length unit for output, e.g. "cm", "m".
    years : float
        Length of period in years.

    Returns
    -------
    tuple[float, float]
        (value_over_period, uncertainty_over_period) in desired_length_unit.
    """
    # Convert dz/dt to desired dz/dt unit per year
    factor = calculate_dzdt_factor(current_unit, f"{desired_length_unit}/yr")

    value_out = value * factor * years
    uncertainty_out = uncertainty * factor * years

    return value_out, uncertainty_out


def volume_from_dzdt(
    value: float,
    uncertainty: float,
    dzdt_unit: str,
    years: float,
    area_m2: float,
) -> tuple[float, float]:
    """
    Convert a subsidence rate (dz/dt) over a given period to volume in million m³,
    including uncertainty.

    Parameters
    ----------
    value : float
        Subsidence rate (e.g.  mm/yr, cm/yr, m/yr).
    uncertainty : float
        Uncertainty in the same dz/dt unit.
    dzdt_unit : str
        Unit of value and uncertainty, e.g. "mm/yr".
    years : float
        Number of years (period).
    area_m2 : float
        Area in square metres. MUST be m².

    Returns
    -------
    tuple[float, float]
        (volume_mln_m3, volume_uncertainty_mln_m3)
    """
    if area_m2 <= 0:
        raise ValueError("area_m2 must be a positive area in m²")

    # dz/dt -> m/yr
    factor = calculate_dzdt_factor(dzdt_unit, "m/yr")

    value_m_yr = value * factor
    uncertainty_m_yr = uncertainty * factor

    # m/yr -> m over periode
    value_m = value_m_yr * years
    uncertainty_m = uncertainty_m_yr * years

    # m * m² -> m³ -> miljoen m³
    volume_mln = (value_m * area_m2) / 1e6
    volume_unc_mln = (uncertainty_m * area_m2) / 1e6

    return volume_mln, volume_unc_mln
