import tomllib
from pathlib import Path

from submon import rasters, stats
from submon.io import export, read

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
    gia_stats = stats.statistics_from_subsidence_rasters(data["gia"])

    combined_raster = rasters.sum_subsidence_rasters(
        data["gia"][0], data["tectonic"][0]
    )
    print(1)
