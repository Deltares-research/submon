from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from submon.rasters import SubsidenceRaster


def statistics_from_subsidence_rasters(
    subsidence_rasters: list[SubsidenceRaster],
) -> list[SubsidenceRaster]:
    """
    Calculate a statistic from a list of SubsidenceRasters.

    """
    pass  # Juliette
