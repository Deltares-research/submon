import tomllib
from pathlib import Path

import rioxarray
import xarray as xr


def load_and_unify_rasters(config_path: str | Path) -> list:
    """Load rasters from disk and combine them into a single xarray Dataset.

    Args:
        raster_paths: List of paths to the raster files.
        raster_names: List of names for the rasters, corresponding to the paths.

    Returns:
        An xarray Dataset containing the loaded rasters.
    """
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    to_unit = config["raster_config"]["unit"]
    to_crs = config["raster_config"]["crs"]

    print(1)


if __name__ == "__main__":
    config_path = Path(
        r"c:\Users\onselen\Development\submon\config\example_config.toml"
    )
    dataset = load_and_unify_rasters(config_path)
