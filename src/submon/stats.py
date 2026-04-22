from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from submon.rasters import SubsidenceRaster

STAT_TO_FUNC = {
    "mean": np.nanmean,
    "min": np.nanmin,
    "max": np.nanmax,
    "median": np.nanmedian,
}


def statistics_from_subsidence_rasters(
    subsidence_rasters: list[SubsidenceRaster],
    stats_to_calculate: list[str] = ["mean", "min", "max"],
) -> xr.Dataset:
    """
    Calculate a statistic from a list of SubsidenceRasters.

    """
    scenario_data = xr.concat(
        [raster.da for raster in subsidence_rasters], dim="scenario"
    )

    results = {}
    for stat in stats_to_calculate:
        if stat not in STAT_TO_FUNC.keys():
            raise ValueError(
                f"Invalid statistic '{stat}' specified. Must be one of {STAT_TO_FUNC.keys()}."
            )

        results[stat] = scenario_data.reduce(STAT_TO_FUNC[stat], dim="scenario")

    return xr.Dataset(
        data_vars=results,
    )
