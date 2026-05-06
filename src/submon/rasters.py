from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import rioxarray as rio
import xarray as xr

from submon import utils

if TYPE_CHECKING:
    from pyproj import CRS


def sum_datasets_per_datavar(ds_l: xr.Dataset, ds_r: xr.Dataset) -> xr.Dataset:
    """
    Parameters
    ----------
    ds_l : xarray.Dataset
        The left-hand side Dataset to combine.
    ds_r : xarray.Dataset
        The right-hand side Dataset to combine.

    Returns
    -------
    xarray.Dataset
        A new Dataset containing the summed data variables.

    Notes
    -----
    This function assumes that the input datasets have already been converted to the same
    CRS and units, and are on the same grid. Data variables that are missing from either
    side are treated as 0 so the other side is preserved.
    """
    all_vars = sorted(set(ds_l.data_vars) & set(ds_r.data_vars))
    data_vars = {name: ds_l.get(name, 0) + ds_r.get(name, 0) for name in all_vars}

    return xr.Dataset(data_vars=data_vars, coords=ds_l.coords, attrs=ds_l.attrs)


def create_grid_from_subsidence_areas(
    subsidence_areas: xr.Dataset, resolution: float
) -> xr.DataArray:
    """
    Create a grid of the specified resolution from a list of subsidence area shapefiles.

    Parameters
    ----------
    subsidence_areas : xr.Dataset
        A Dataset containing subsidence area polygons.
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
