import pytest

from submon.units import calculate_dzdt_factor, parse_dzdt_units


def test_parse_dzdt_units():
    assert parse_dzdt_units("mm/yr") == ("mm", "yr")
    assert parse_dzdt_units("m/s") == ("m", "s")
    assert parse_dzdt_units("cm/min") == ("cm", "min")

    with pytest.raises(ValueError):
        parse_dzdt_units("invalid_unit")


@pytest.mark.unittest
def test_calculate_dzdt_factor():
    # Test mm/yr to m/yr
    factor = calculate_dzdt_factor("mm/yr", "m/yr")
    assert factor == 0.001

    # Test m/s to mm/d
    factor = calculate_dzdt_factor("m/s", "mm/d")
    assert factor == 1000 * 86400

    # Test cm/min to dm/h
    factor = calculate_dzdt_factor("cm/min", "dm/h")
    assert factor == (10 / 100) * (60 / 3600)

    # Test invalid units
    with pytest.raises(ValueError):
        calculate_dzdt_factor("invalid_unit", "m/yr")

    with pytest.raises(ValueError):
        calculate_dzdt_factor("mm/yr", "invalid_unit")
