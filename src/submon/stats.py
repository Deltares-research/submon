from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from submon.rasters import SubsidenceRaster


def statistics_from_subsidence_rasters(
    subsidence_rasters: list[SubsidenceRaster],
) -> list[SubsidenceRaster]:
    """
    Calculate a statistic from a list of SubsidenceRasters.

    """
    test = xr.concat([raster.da for raster in subsidence_rasters], "new_dim")
    test = test.mean("new_dim")

    return test