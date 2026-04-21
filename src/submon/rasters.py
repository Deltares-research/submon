from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import result

import geopandas as gpd
import numpy as np
import rioxarray as rio
import xarray as xr

from submon import utils

if TYPE_CHECKING:
    from pyproj import CRS


@dataclass
class SubsidenceRaster:
    da: xr.DataArray
    source_path: str | Path | list[str | Path]
    subsidence_type: str
    statistic_type: str | None
    original_crs: str | int | CRS
    converted_crs: str | int | CRS
    original_units: str
    converted_units: str

    def __repr__(self):
        if isinstance(self.source_path, list):
            source_paths = [Path(p).stem for p in self.source_path]
            source_paths_str = ", ".join(source_paths)
        else:
            source_paths_str = Path(self.source_path).stem

        return f"SubsidenceRaster(source_path={source_paths_str}, subsidence_type={self.subsidence_type}, statistic_type={self.statistic_type}, original_crs={self.original_crs}, converted_crs={self.converted_crs}, original_units={self.original_units}, converted_units={self.converted_units})"


def sum_subsidence_rasters(
    raster_l: dict[str, xr.DataArray], raster_r: dict[str, xr.DataArray]
) -> dict[str, xr.DataArray]:
    """
    Combine two dictionaries of DataArrays into a single one by summing their DataArrays.

    Parameters
    ----------
    raster_l : dict[str, xr.DataArray]
        The left-hand side dictionary of DataArrays to combine.
    raster_r : dict[str, xr.DataArray]
        The right-hand side dictionary of DataArrays to combine.

    Returns
    -------
    SubsidenceRaster
        A new SubsidenceRaster containing the combined DataArray and metadata from the
        input rasters.

    Notes
    -----
    This function assumes that the input rasters have already been converted to the same
    CRS and units. The resulting DataArray is the sum of the two input DataArrays, and
    the metadata is taken from the left-hand side raster.
    """
    combined = {}
    for key in raster_l:
        combined[key] = raster_l[key] + raster_r[key]
    return combined


def create_grid_from_subsidence_areas(
    subsidence_areas: gpd.GeoDataFrame, resolution: float
) -> xr.DataArray:
    """
    Create a grid of the specified resolution from a list of subsidence area shapefiles.

    Parameters
    ----------
    subsidence_areas : gpd.GeoDataFrame
        A GeoDataFrame containing subsidence area polygons.
    resolution : float
        The desired resolution of the output grid in the same units as the CRS of the
        input shapefiles.

    Returns
    -------
    xarray.DataArray

    """
    xmin, ymin, xmax, ymax = subsidence_areas.total_bounds
    x_coords = np.arange(xmin, xmax + resolution, resolution)
    y_coords = np.arange(ymin, ymax + resolution, resolution)

    da = xr.DataArray(
        np.zeros((len(y_coords), len(x_coords))),
        coords={"y": y_coords, "x": x_coords},
        dims=("y", "x"),
    )
    da = utils.set_da_attributes(
        da,
        transform=(
            resolution,
            0.0,
            x_coords[0] - 0.5 * resolution,
            0.0,
            -resolution,
            y_coords[0] + 0.5 * resolution,
        ),
        res=(resolution, resolution),
        is_tiled=0,
        scales=(1.0,),
        offset=(0.0,),
        nodatavals=(np.nan,),
    )
    da.rio.write_crs(subsidence_areas.crs, inplace=True)

    return da
