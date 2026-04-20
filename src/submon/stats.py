from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from submon.rasters import SubsidenceRaster


def statistics_from_subsidence_rasters(
    subsidence_rasters: list[SubsidenceRaster],
) -> list[SubsidenceRaster]:
    from submon.rasters import SubsidenceRaster

    """
    Calculate a statistic from a list of SubsidenceRasters.

    """
    scenario_data = xr.concat(
        [raster.da for raster in subsidence_rasters], dim="scenario"
    )
    mean_da = scenario_data.mean("scenario")
    min_da = scenario_data.min("scenario")
    max_da = scenario_data.max("scenario")

    return mean_da, min_da, max_da
