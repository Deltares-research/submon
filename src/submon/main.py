import tomllib
from pathlib import Path

import pandas as pd
import xarray as xr

from submon import rasters, stats, utils
from submon.io import read
from submon.io.export import to_geotiff

"""
Compute subsidence components (GIA, Tectonics, Combined, Mining) and total subsidence
with mining uncertainty. Export results to NetCDF and GeoTIFF,
and calculate zonal statistics for specified areas.
"""

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

    # GIA
    # statistics (mean, max en min) from the statistics_from_subsidence_raster extraction
    gia_stats = stats.statistics_from_dataarrays(data["gia"], config["gia"]["stats"])

    # Tectonic
    tect_stats = xr.Dataset(data_vars={x.statistic_type: x for x in data["tectonic"]})

    # Combined (GIA + Tectonics)
    combined_stats = rasters.sum_datasets_per_datavar(gia_stats, tect_stats)

    # mining data
    mining_da = data["mining"][0]

    # mining with uncertainty
    mining_last30 = stats.predefined_statistics(
        mining_da, {"mean": 1.0, "min": 0.75, "max": 1.25}
    )
    mining_next30 = stats.predefined_statistics(
        mining_da, {"mean": 1.0, "min": 0.50, "max": 1.50}
    )

    # total subsidence with mining uncertainty
    subsidence_last30 = rasters.sum_datasets_per_datavar(combined_stats, mining_last30)
    subsidence_next30 = rasters.sum_datasets_per_datavar(combined_stats, mining_next30)

    # Export static subsidence components (no period dimension) to GeoTIFF
    for key, obj in [
        ("GIA", gia_stats),
        ("Tectonic", tect_stats),
        ("Combined", combined_stats),
        ("Mining", mining_da),
        ("Total_Last30", subsidence_last30),
        ("Total_Next30", subsidence_next30),
    ]:
        out_dir = (
            Path(config["output"]["base"]) / config["output"]["paths"][key.lower()]
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        to_geotiff(
            obj,
            out_dir,
            compress=True,
            prefix=f"{key.lower()}_",
        )

    # empty list for every investageted area, to be filled with zonal statistics and volumes
    rows = []
    # loop over investigated areas and calculate zonal statistics for each subsidence component, including mining uncertainty scenarios, and calculate volumes based on mean subsidence and area. Save results to Excel.
    for _, row in subsidence_areas.iterrows():
        geom = row.geometry
        gebied = row["Gebied"]
        area = geom.area

        # Clip all datasets to the geometry
        gia_clipped = gia_stats.rio.clip(
            [geom], crs=subsidence_areas.crs, drop=True, all_touched=True
        )
        tect_clipped = tect_stats.rio.clip(
            [geom], crs=subsidence_areas.crs, drop=True, all_touched=True
        )
        combined_clipped = combined_stats.rio.clip(
            [geom], crs=subsidence_areas.crs, drop=True, all_touched=True
        )
        mining_last30_clipped = mining_last30.rio.clip(
            [geom], crs=subsidence_areas.crs, drop=True, all_touched=True
        )
        mining_next30_clipped = mining_next30.rio.clip(
            [geom], crs=subsidence_areas.crs, drop=True, all_touched=True
        )
        total_last30_clipped = subsidence_last30.rio.clip(
            [geom], crs=subsidence_areas.crs, drop=True, all_touched=True
        )
        total_next30_clipped = subsidence_next30.rio.clip(
            [geom], crs=subsidence_areas.crs, drop=True, all_touched=True
        )

        # Compute mean values for each clipped dataset (returns dict like {'mean': 0.5, 'min': 0.2, 'max': 0.7})
        gia_values = stats.mean_value_of_dataset_vars(gia_clipped)
        tect_values = stats.mean_value_of_dataset_vars(tect_clipped)
        combined_values = stats.mean_value_of_dataset_vars(combined_clipped)
        mining_last30_values = stats.mean_value_of_dataset_vars(mining_last30_clipped)
        mining_next30_values = stats.mean_value_of_dataset_vars(mining_next30_clipped)
        total_last30_values = stats.mean_value_of_dataset_vars(total_last30_clipped)
        total_next30_values = stats.mean_value_of_dataset_vars(total_next30_clipped)

        # Append results to rows list, including formatted statistics and calculated volumes based on mean subsidence and area
        rows.append(
            {
                "Gebied": gebied,
                "GIA": utils.format_stats_as_text(gia_values),
                "Tect": utils.format_stats_as_text(tect_values),
                "CombinedGeology": utils.format_stats_as_text(combined_values),
                "Mining_last30": utils.format_stats_as_text(mining_last30_values),
                "Total_last30": utils.format_stats_as_text(total_last30_values),
                "Mining_next30": utils.format_stats_as_text(mining_next30_values),
                "Total_next30": utils.format_stats_as_text(total_next30_values),
                "Vol_Mm3_GIA": stats.volume(gia_values["mean"], area),
                "Vol_Mm3_Tect": stats.volume(tect_values["mean"], area),
                "Vol_Mm3_CombinedGeology": stats.volume(combined_values["mean"], area),
                "Vol_Mm3_Mining_last30": stats.volume(
                    mining_last30_values["mean"], area
                ),
                "Vol_Mm3_Total_last30": stats.volume(total_last30_values["mean"], area),
                "Vol_Mm3_Total_next30": stats.volume(total_next30_values["mean"], area),
            }
        )
    # Save results to Excel
    df = pd.DataFrame(rows)
    df.to_excel(
        Path(config["output"]["base"]) / config["output"]["files"]["excel"],
        index=False,
    )

    print("Done!")
