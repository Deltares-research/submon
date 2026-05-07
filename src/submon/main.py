import logging
import tomllib
from pathlib import Path

import pandas as pd
import xarray as xr

from submon import rasters, stats, units, utils
from submon.io import read
from submon.io.export import to_geotiff

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


"""
Compute subsidence components (GIA, Tectonics, Combined, Mining) and total subsidence
with mining uncertainty. Export results to NetCDF and GeoTIFF,
and calculate zonal statistics for specified areas.
"""

if __name__ == "__main__":
    logger.info("Starting subsidence monitoring workflow...")
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
    logger.info("Loading and unifying subsidence rasters...")
    data = read.load_subsidence_rasters(config, target_grid)

    # GIA
    # statistics (mean, max en min) from the statistics_from_subsidence_raster extraction
    logger.info("Processing GIA")
    gia_stats = stats.statistics_from_dataarrays(
        data["gia"], config["gia"]["stats"], invert_min_max=True
    )

    # Tectonic
    logger.info("Processing Tectonic")
    tect_stats = xr.Dataset(data_vars={x.statistic_type: x for x in data["tectonic"]})

    # Combined (GIA + Tectonics)
    logger.info("Processing geological subsidence (GIA + Tectonics)")
    geological_stats = rasters.sum_datasets_per_datavar(gia_stats, tect_stats)

    # mining data
    logger.info("Processing mining data")
    mining_da = data["mining"][0]

    # mining with uncertainty
    mining_last30 = stats.predefined_statistics(
        mining_da, {"mean": 1.0, "min": 0.75, "max": 1.25}
    )
    mining_next30 = stats.predefined_statistics(
        mining_da, {"mean": 1.0, "min": 0.50, "max": 1.50}
    )

    # total subsidence with mining uncertainty
    subsidence_last30 = rasters.sum_datasets_per_datavar(
        geological_stats, mining_last30
    )
    subsidence_next30 = rasters.sum_datasets_per_datavar(
        geological_stats, mining_next30
    )

    # Export static subsidence components (no period dimension) to GeoTIFF
    logger.info("Exporting static subsidence components to GeoTIFF...")
    Path(config["output_paths"]["base"]).mkdir(parents=True, exist_ok=True)

    for key, obj in [
        ("gia", gia_stats["mean"]),
        ("tectonic", tect_stats["mean"]),
        ("geological", geological_stats["mean"]),
        ("mining", mining_da),
        ("total_last30", subsidence_last30["mean"]),
        ("total_next30", subsidence_next30["mean"]),
    ]:
        out_file = Path(config["output_paths"]["base"]) / config["output_paths"][key]

        to_geotiff(
            obj,
            out_file,
            compress=True,
        )

    logger.info("Exports complete.")
    logger.info("Calculating zonal statistics for subsidence areas...")

    # empty list for every investageted area, to be filled with zonal statistics and volumes
    rows = []
    # loop over investigated areas and calculate zonal statistics for each subsidence component, including mining uncertainty scenarios, and calculate volumes based on mean subsidence and area. Save results to Excel.
    for _, row in subsidence_areas.iterrows():
        geom = row.geometry
        gebied = row["Gebied"]
        area = geom.area

        logger.info(f"Processing area '{gebied}'")
        # Clip all datasets to the geometry
        # GIA
        gia_clipped = gia_stats.rio.clip(
            [geom], crs=subsidence_areas.crs, drop=True, all_touched=True
        )
        gia_subsidence = gia_clipped["mean"].mean().values
        gia_uncertainty = stats.calculate_uncertainty(gia_clipped)

        # Tectonic
        tect_clipped = tect_stats.rio.clip(
            [geom], crs=subsidence_areas.crs, drop=True, all_touched=True
        )
        tect_subsidence = tect_clipped["mean"].mean().values
        tect_uncertainty = stats.calculate_uncertainty(tect_clipped)

        # Combined geological
        geological_clipped = geological_stats.rio.clip(
            [geom], crs=subsidence_areas.crs, drop=True, all_touched=True
        )
        geological_subsidence = geological_clipped["mean"].mean().values
        geological_uncertainty = stats.combine_uncertainties(
            gia_uncertainty, tect_uncertainty
        )

        # Mining
        mining_last30_clipped = mining_last30.rio.clip(
            [geom], crs=subsidence_areas.crs, drop=True, all_touched=True
        )
        mining_last30_subsidence = mining_last30_clipped["mean"].mean().values
        mining_last30_uncertainty = stats.calculate_uncertainty(mining_last30_clipped)

        mining_next30_clipped = mining_next30.rio.clip(
            [geom], crs=subsidence_areas.crs, drop=True, all_touched=True
        )
        mining_next30_subsidence = mining_next30_clipped["mean"].mean().values
        mining_next30_uncertainty = stats.calculate_uncertainty(mining_next30_clipped)

        # Grand total with mining
        total_last30_clipped = subsidence_last30.rio.clip(
            [geom], crs=subsidence_areas.crs, drop=True, all_touched=True
        )
        total_last30_subsidence = total_last30_clipped["mean"].mean().values
        total_last30_uncertainty = stats.combine_uncertainties(
            geological_uncertainty, mining_last30_uncertainty
        )

        total_next30_clipped = subsidence_next30.rio.clip(
            [geom], crs=subsidence_areas.crs, drop=True, all_touched=True
        )
        total_next30_subsidence = total_next30_clipped["mean"].mean().values
        total_next30_uncertainty = stats.combine_uncertainties(
            geological_uncertainty, mining_next30_uncertainty
        )

        # Append results to rows list, including formatted statistics and calculated volumes based on mean subsidence and area
        last_x_years = config["output_config"]["last_x_years"]
        next_x_years = config["output_config"]["next_x_years"]
        current_unit = config["output_config"]["unit"]

        rows.append(
            {
                "Gebied": gebied,
                "GIA": utils.format_for_output_table(
                    gia_subsidence, gia_uncertainty, current_unit, "mm/yr"
                ),
                "Tect": utils.format_for_output_table(
                    tect_subsidence, tect_uncertainty, current_unit, "mm/yr"
                ),
                "CombinedGeology": utils.format_for_output_table(
                    geological_subsidence, geological_uncertainty, current_unit, "mm/yr"
                ),
                # TODO verder afmaken
                "Mining_last30": utils.format_for_output_table(
                    mining_last30_subsidence,
                    mining_last30_uncertainty,
                    current_unit,
                    f"cm/{last_x_years}yr",
                ),
                "Total_last30": utils.format_for_output_table(
                    total_last30_subsidence, total_last30_uncertainty
                ),
                f"Mining_next{next_x_years}": utils.format_for_output_table(
                    mining_next30_subsidence,
                    mining_next30_uncertainty,
                ),
                "Total_next30": utils.format_for_output_table(
                    total_next30_subsidence, total_next30_uncertainty
                ),
                "Vol_Mm3_GIA": stats.volume(gia_subsidence, area),
                "Vol_Mm3_Tect": stats.volume(tect_subsidence, area),
                "Vol_Mm3_CombinedGeology": stats.volume(geological_subsidence, area),
                "Vol_Mm3_Mining_last30": stats.volume(mining_last30_subsidence, area),
                "Vol_Mm3_Total_last30": stats.volume(total_last30_subsidence, area),
                "Vol_Mm3_Total_next30": stats.volume(total_next30_subsidence, area),
            }
        )
    # Save results to Excel
    df = pd.DataFrame(rows)
    logger.info("Saving zonal statistics and volumes to output Excel...")
    df.to_excel(
        Path(config["output_paths"]["base"]) / config["output_paths"]["output_excel"],
        index=False,
    )

    logger.info("Done!")
