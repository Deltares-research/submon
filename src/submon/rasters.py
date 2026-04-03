from pathlib import Path

import rioxarray
import xarray as xr


def load_rasters(
    raster_paths: list[Path],
    raster_names: list[str],
) -> xr.Dataset:
    """Load rasters from disk and combine them into a single xarray Dataset.

    Args:
        raster_paths: List of paths to the raster files.
        raster_names: List of names for the rasters, corresponding to the paths.

    Returns:
        An xarray Dataset containing the loaded rasters.
    """
    datasets = []
    for path, name in zip(raster_paths, raster_names):
        raster = rioxarray.open_rasterio(path)
        raster = raster.rename(name)
        datasets.append(raster)

    return xr.merge(datasets)
