from dataclasses import dataclass
from pathlib import Path

import rioxarray
import xarray as xr
from pyproj import CRS


@dataclass
class SubsidenceRaster:
    da: xr.DataArray
    source_path: str | Path
    subsidence_type: str
    statistic_type: str | None
    original_crs: str | int | CRS
    converted_crs: str | int | CRS
    original_units: str
    converted_units: str

    def __repr__(self):
        return f"SubsidenceRaster(source_path={Path(self.source_path).stem}, subsidence_type={self.subsidence_type}, statistic_type={self.statistic_type}, original_crs={self.original_crs}, converted_crs={self.converted_crs}, original_units={self.original_units}, converted_units={self.converted_units})"


def statistics_from_subsidence_rasters(
    subsidence_rasters: list[SubsidenceRaster],
) -> list[SubsidenceRaster]:
    """
    Calculate a statistic from a list of SubsidenceRasters.

    """
    pass  # Juliette
