from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import rioxarray
import xarray as xr
from pyproj import CRS

from submon import rasters, units
from submon.io import xyz

if TYPE_CHECKING:
    from submon.rasters import SubsidenceRaster


def load_subsidence_areas(config: dict) -> gpd.GeoDataFrame:
    """
    Load subsidence areas from disk based on a configuration dictionary.

    Parameters
    ----------
    config : dict
        A dictionary containing the subsidence area configuration, including paths and names.
        Expected to contain a 'subsidence_areas' key with a list of area configurations.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the loaded subsidence areas with their metadata.

    Raises
    ------
    ValueError
        If the subsidence areas file does not have a defined CRS.

    """
    areas = gpd.read_file(config["investigated_areas"]["path"])

    if not areas.crs:
        raise ValueError("The subsidence areas file must have a defined CRS.")

    areas = areas.to_crs(config["output_config"]["epsg"])

    return areas


def load_subsidence_rasters(
    config: dict, target_grid: xr.DataArray
) -> dict[SubsidenceRaster]:
    """
    Load subsidence rasters from disk based on a configuration dictionary, applying
    coordinate reference system and unit conversions as specified.

    Parameters
    ----------
    config : dict
        A dictionary containing the raster configuration, including paths and names.
        Expected to contain keys for subsidence sources ('gia', 'tectonic', 'mining')
        and a 'raster_config' key with unit and epsg conversion settings.
    target_grid : xr.DataArray
        An xarray DataArray representing the target grid to which rasters should be
        reprojected.

    Returns
    -------
    dict
        A dictionary where keys are subsidence source types ('gia', 'tectonic', 'mining')
        and values are lists of SubsidenceRaster objects containing the loaded rasters
        with their metadata.

    Notes
    -----
    The function filters configuration keys to only process recognized subsidence sources
    (gia, tectonic, and mining). Each raster is loaded, converted to the target units and
    CRS, and wrapped in a SubsidenceRaster object with metadata.
    """
    subsidence_sources = [
        s for s in config.keys() if s in ("gia", "tectonic", "mining")
    ]

    data = {}

    for source in subsidence_sources:
        data[source] = []
        for raster_info in config[source]["rasters"]:
            raster = load_and_convert_raster(
                raster_info["path"],
                raster_info["units"],
                config["output_config"]["unit"],
                raster_info["epsg"],
                config["output_config"]["epsg"],
                target_grid=target_grid,
                subsidence_positive=raster_info["subsidence_positive"],
                **raster_info.get("reader_kwargs", {}),
            )

            # Add metadata attributes to the DataArray for traceability
            raster.attrs["source_path"] = raster_info["path"]
            raster.attrs["subsidence_type"] = source
            raster.attrs["statistic_type"] = raster_info.get("stat", None)
            raster.attrs["original_crs"] = CRS.from_user_input(raster_info["epsg"])
            raster.attrs["converted_crs"] = CRS.from_user_input(
                config["output_config"]["epsg"]
            )
            raster.attrs["original_units"] = raster_info["units"]
            raster.attrs["converted_units"] = config["output_config"]["unit"]

            data[source].append(raster)

    return data


def load_and_convert_raster(
    path: str | Path,
    dzdt_from: str,
    dzdt_to: str,
    from_epsg: int | str | CRS,
    to_epsg: int | str | CRS,
    target_grid: xr.DataArray = None,
    subsidence_positive: bool = True,
    **kwargs,
) -> xr.DataArray:
    """
    Load and convert a raster file with coordinate and unit transformations.
    This function loads a raster from disk (supporting various formats), reprojects
    it to a target coordinate reference system, and converts vertical change rate
    units as specified.

    Parameters
    ----------
    path : str or Path
        Path to the raster file. Supports .xyz, .txt, .csv, .pts, or standard
        raster formats (e.g., GeoTIFF, esri-style folder).
    dzdt_from : str
        Source unit for vertical change rate (e.g., 'm/year').
    dzdt_to : str
        Target unit for vertical change rate (e.g., 'mm/year').
    from_epsg : int, str, or CRS
        Source coordinate reference system as EPSG code or CRS object.
    to_epsg : int, str, or CRS
        Target coordinate reference system as EPSG code or CRS object.
    **kwargs
        Additional keyword arguments passed to the raster reading function
        (rioxarray.open_rasterio or xyz.read_xyz).

    Returns
    -------
    xr.DataArray
        Loaded raster as an xarray DataArray, reprojected to target CRS and
        scaled to target units. Single-band rasters are squeezed to remove
        degenerate dimensions.
    """

    if Path(path).suffix in [".xyz", ".txt", ".csv", ".pts"]:
        da = xyz.read_xyz(
            path,
            gridded=kwargs.get("gridded", True),
        )
    else:
        da = rioxarray.open_rasterio(path, **kwargs).squeeze()

    da.rio.write_crs(from_epsg, inplace=True)
    if CRS.from_user_input(from_epsg) != CRS.from_user_input(to_epsg):
        da = da.rio.reproject(to_epsg)

    if target_grid is not None:
        da = da.rio.reproject_match(target_grid)

    factor = units.calculate_dzdt_factor(dzdt_from, dzdt_to)
    da *= factor

    if subsidence_positive:
        da *= -1

    if "_FillValue" in da.attrs:
        da = da.where(da != da.attrs["_FillValue"], other=np.nan)
        da = da.rio.write_nodata(np.nan, inplace=True)

    return da
