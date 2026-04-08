import tomllib
from pathlib import Path

import rioxarray
import xarray as xr

from submon import rasters, units
from submon.io import xyz


def load_submon_rasters(config: dict) -> xr.Dataset:
    """Load rasters from disk and combine them into a single xarray Dataset.

    Args:
        config: A dictionary containing the raster configuration, including paths and names.
    Returns:
        An xarray Dataset containing the loaded rasters.
    """
    subsidence_sources = [
        s for s in config.keys() if s in ("gia", "tectonic", "mining")
    ]

    data = []

    for source in subsidence_sources:
        for raster_info in config[source]["rasters"]:
            raster = load_and_convert_raster(
                raster_info["path"],
                raster_info["units"],
                config["raster_config"]["unit"],
                raster_info["epsg"],
                config["raster_config"]["epsg"],
                **raster_info.get("reader_kwargs", {}),
            )
            raster_object = rasters.SubsidenceRaster(
                da=raster,
                source_path=raster_info["path"],
                subsidence_type=source,
                original_crs=raster_info["epsg"],
                converted_crs=config["raster_config"]["epsg"],
                original_units=raster_info["units"],
                converted_units=config["raster_config"]["unit"],
            )
            data.append(raster_object)

    return data


def load_and_convert_raster(
    path: str | Path,
    dzdt_from: str,
    dzdt_to: str,
    from_epsg: int,
    to_epsg: int,
    **kwargs,
) -> list:
    """Load rasters from disk and combine them into a single xarray Dataset.

    Args:
        raster_paths: List of paths to the raster files.
        raster_names: List of names for the rasters, corresponding to the paths.

    Returns:
        An xarray Dataset containing the loaded rasters.
    """

    if Path(path).suffix in [".xyz", ".txt", ".csv", ".pts"]:
        da = xyz.read_xyz(
            path,
            gridded=kwargs.get("gridded", True),
        )
    else:
        da = rioxarray.open_rasterio(path, **kwargs)

    da.rio.write_crs(from_epsg, inplace=True)
    da = da.rio.reproject(to_epsg)
    factor = units.calculate_dzdt_factor(dzdt_from, dzdt_to)
    da *= factor

    return da


if __name__ == "__main__":
    config_path = Path(
        r"c:\Users\onselen\Development\submon\config\example_config.toml"
    )
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    data = load_submon_rasters(config)
