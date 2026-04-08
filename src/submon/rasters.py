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
    original_crs: str | int | CRS
    converted_crs: str | int | CRS
    original_units: str
    converted_units: str
