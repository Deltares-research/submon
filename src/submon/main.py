import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from submon import rasters, stats
from submon.io import export, read
from submon.io.export import to_geotiff, to_nc

# output paths
output_path = Path(r"C:\Projecten\bodemdaling\output")
gia_dir = Path(output_path / "gia")
tect_dir = output_path / "tectonic"
combined_dir = output_path / "subsidence_combined"
mining_dir = output_path / "mining"
total_dir = output_path / "subsidence_total"

gia_dir.mkdir(parents=True, exist_ok=True)
tect_dir.mkdir(parents=True, exist_ok=True)
combined_dir.mkdir(parents=True, exist_ok=True)
mining_dir.mkdir(parents=True, exist_ok=True)
total_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    config_path = Path(__file__).parents[2] / Path("config/example_config.toml")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    # Load subsidence areas from config
    subsidence_areas = read.load_subsidence_areas(config)

    # Create target grid from subsidence areas and resolution specified in config
    target_grid = rasters.create_grid_from_subsidence_areas(
        subsidence_areas, config["output_config"]["resolution"]
    )

    # Load subsidence rasters from config, reprojecting and matching them to the target grid
    data = read.load_subsidence_rasters(config, target_grid)

    # Hieronder verder werken aan berekeningen...
    # statistics (mean, max en min) from the statistics_from_subsidence_raster extraction
    gia_stats = stats.statistics_from_subsidence_rasters(
        data["gia"], config["gia"]["stats"]
    )

    # Tectonic subsidence statistics
    tect_stats = xr.Dataset(
        data_vars={x.statistic_type: x.da for x in data["tectonic"]}
    )
    # GIA and Tectonic combined geological subsidence (GIA + Tectonics)
    combined_stats = rasters.sum_subsidence_rasters(gia_stats, tect_stats)

    # export of GIA, Tectonics and combined to geotiff files
    exports = [
        (gia_stats, gia_dir / "gia.nc"),
        (tect_stats, tect_dir / "tect.nc"),
        (combined_stats, combined_dir / "subsidence_combined.nc"),
    ]

    for ds, out_file in exports:
        to_nc(ds, out_file, compress=True)

    # mining data
    mining_stats = xr.Dataset(
        data_vars={x.statistic_type: x.da for x in data["mining"]}
    )

    print("Done!")
