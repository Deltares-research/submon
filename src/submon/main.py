import logging
import sys
import tomllib
from pathlib import Path

import geopandas as gpd
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
with mining uncertainty. Export results to GeoTIFF,
and calculate zonal statistics for specified areas.
"""

if __name__ == "__main__":
    logger.info("Starting subsidence monitoring workflow...")
    try:
        config_path = Path(sys.argv[1])
    except IndexError:
        config_path = Path(__file__).parents[2] / Path("config/config.toml")
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
    mining_da_last_x_years = data["mining"][0]
    mining_da_next_x_years = data["mining"][1]

    # mining with uncertainty
    mining_last_x_years = stats.predefined_statistics(
        mining_da_last_x_years, {"mean": 1.0, "min": 0.75, "max": 1.25}
    )
    mining_next_x_years = stats.predefined_statistics(
        mining_da_next_x_years, {"mean": 1.0, "min": 0.50, "max": 1.50}
    )

    # total subsidence with mining uncertainty
    subsidence_last_x_years = rasters.sum_datasets_per_datavar(
        geological_stats, mining_last_x_years
    )
    subsidence_next_x_years = rasters.sum_datasets_per_datavar(
        geological_stats, mining_next_x_years
    )

    # Append results to rows list, including formatted statistics and calculated volumes based on mean subsidence and area
    last_x_years = config["output_config"]["last_x_years"]
    next_x_years = config["output_config"]["next_x_years"]
    current_unit = config["output_config"]["unit"]

    # Export static subsidence components (no period dimension) to GeoTIFF
    logger.info("Exporting static subsidence components to GeoTIFF...")
    Path(config["output_paths"]["base"]).mkdir(parents=True, exist_ok=True)

    for key, obj in [
        ("gia", gia_stats),
        ("tectonic", tect_stats),
        ("geological", geological_stats),
        ("mining_last", mining_last_x_years),
        ("mining_next", mining_next_x_years),
        ("total_subsidence_last", subsidence_last_x_years),
        ("total_subsidence_next", subsidence_next_x_years),
    ]:
        out_dir = Path(config["output_paths"]["base"]) / config["output_paths"][key]
        out_dir.mkdir(parents=True, exist_ok=True)

        for datavar in obj.data_vars:
            to_geotiff(
                obj[datavar],
                out_dir / f"{key}_{datavar}.tif",
                compress=True,
            )

    logger.info("Exports complete.")
    logger.info("Calculating zonal statistics for subsidence areas...")

    # empty list for every investageted area, to be filled with zonal statistics and volumes
    rows = []
    shape_rows = []
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
        mining_last_x_years_clipped = mining_last_x_years.rio.clip(
            [geom], crs=subsidence_areas.crs, drop=True, all_touched=True
        )
        mining_last_x_years_subsidence = (
            mining_last_x_years_clipped["mean"].mean().values
        )
        mining_last_x_years_uncertainty = stats.calculate_uncertainty(
            mining_last_x_years_clipped
        )

        mining_next_x_years_clipped = mining_next_x_years.rio.clip(
            [geom], crs=subsidence_areas.crs, drop=True, all_touched=True
        )
        mining_next_x_years_subsidence = (
            mining_next_x_years_clipped["mean"].mean().values
        )
        mining_next_x_years_uncertainty = stats.calculate_uncertainty(
            mining_next_x_years_clipped
        )

        # Grand total with mining
        total_last_x_years_clipped = subsidence_last_x_years.rio.clip(
            [geom], crs=subsidence_areas.crs, drop=True, all_touched=True
        )
        total_last_x_years_subsidence = total_last_x_years_clipped["mean"].mean().values
        total_last_x_years_uncertainty = stats.combine_uncertainties(
            geological_uncertainty, mining_last_x_years_uncertainty
        )

        total_next_x_years_clipped = subsidence_next_x_years.rio.clip(
            [geom], crs=subsidence_areas.crs, drop=True, all_touched=True
        )
        total_next_x_years_subsidence = total_next_x_years_clipped["mean"].mean().values
        total_next_x_years_uncertainty = stats.combine_uncertainties(
            geological_uncertainty, mining_next_x_years_uncertainty
        )

        # Append all results to rows list for Excel and shape_rows for shapefile
        rows.append(
            {
                "Gebied": gebied,
                "Oppervlakte_m2": area,
                "GIA_mm/yr": utils.format_for_output_table(
                    gia_subsidence, gia_uncertainty, current_unit, "mm/yr"
                ),
                "Tectonic_mm/yr": utils.format_for_output_table(
                    tect_subsidence, tect_uncertainty, current_unit, "mm/yr"
                ),
                "Geological_subsidence_mm/yr": utils.format_for_output_table(
                    geological_subsidence, geological_uncertainty, current_unit, "mm/yr"
                ),
                f"GIA_last_{last_x_years}_years_cm": utils.format_for_output_table(
                    gia_subsidence,
                    gia_uncertainty,
                    current_unit,
                    f"cm/{last_x_years}yr",
                    nd=2,
                ),
                f"GIA_next_{next_x_years}_years_cm": utils.format_for_output_table(
                    gia_subsidence,
                    gia_uncertainty,
                    current_unit,
                    f"cm/{next_x_years}yr",
                    nd=2,
                ),
                f"Tectonic_last_{last_x_years}_years_cm": utils.format_for_output_table(
                    tect_subsidence,
                    tect_uncertainty,
                    current_unit,
                    f"cm/{last_x_years}yr",
                    nd=2,
                ),
                f"Tectonic_next_{next_x_years}_years_cm": utils.format_for_output_table(
                    tect_subsidence,
                    tect_uncertainty,
                    current_unit,
                    f"cm/{next_x_years}yr",
                    nd=2,
                ),
                f"Geological_last_{last_x_years}_years_cm": utils.format_for_output_table(
                    geological_subsidence,
                    geological_uncertainty,
                    current_unit,
                    f"cm/{last_x_years}yr",
                    nd=2,
                ),
                f"Geological_next_{next_x_years}_years_cm": utils.format_for_output_table(
                    geological_subsidence,
                    geological_uncertainty,
                    current_unit,
                    f"cm/{next_x_years}yr",
                    nd=2,
                ),
                f"Mining_last_{last_x_years}_years_cm": utils.format_for_output_table(
                    mining_last_x_years_subsidence,
                    mining_last_x_years_uncertainty,
                    current_unit,
                    f"cm/{last_x_years}yr",
                    nd=2,
                ),
                f"Mining_next_{next_x_years}_years_cm": utils.format_for_output_table(
                    mining_next_x_years_subsidence,
                    mining_next_x_years_uncertainty,
                    current_unit,
                    f"cm/{next_x_years}yr",
                    nd=2,
                ),
                f"Total_last_{last_x_years}_years_cm": utils.format_for_output_table(
                    total_last_x_years_subsidence,
                    total_last_x_years_uncertainty,
                    current_unit,
                    f"cm/{last_x_years}yr",
                    nd=2,
                ),
                f"Total_next_{next_x_years}_years_cm": utils.format_for_output_table(
                    total_next_x_years_subsidence,
                    total_next_x_years_uncertainty,
                    current_unit,
                    f"cm/{next_x_years}yr",
                    nd=2,
                ),
                f"Total_subsidence_last_{last_x_years}_cm/yr": utils.format_for_output_table(
                    total_last_x_years_subsidence,
                    total_last_x_years_uncertainty,
                    current_unit,
                    "cm/yr",
                ),
                f"Total_subsidence_next_{next_x_years}_cm/yr": utils.format_for_output_table(
                    total_next_x_years_subsidence,
                    total_next_x_years_uncertainty,
                    current_unit,
                    "cm/yr",
                ),
                f"Geological_Volume_last_{last_x_years}_Mm3": utils.format_for_output_table(
                    geological_subsidence * area,
                    geological_uncertainty * area,
                    current_unit,
                    f"Mm/{last_x_years}yr",
                ),
                f"Geological_Volume_next_{next_x_years}_Mm3": utils.format_for_output_table(
                    geological_subsidence * area,
                    geological_uncertainty * area,
                    current_unit,
                    f"Mm/{next_x_years}yr",
                ),
                f"Mining_Volume_last_{last_x_years}_Mm3": utils.format_for_output_table(
                    mining_last_x_years_subsidence * area,
                    mining_last_x_years_uncertainty * area,
                    current_unit,
                    f"Mm/{last_x_years}yr",
                ),
                f"Mining_Volume_next_{next_x_years}_Mm3": utils.format_for_output_table(
                    mining_next_x_years_subsidence * area,
                    mining_next_x_years_uncertainty * area,
                    current_unit,
                    f"Mm/{next_x_years}yr",
                ),
                f"Total_Volume_last_{last_x_years}_Mm3": utils.format_for_output_table(
                    total_last_x_years_subsidence * area,
                    total_last_x_years_uncertainty * area,
                    current_unit,
                    f"Mm/{last_x_years}yr",
                ),
                f"Total_Volume_next_{next_x_years}_Mm3": utils.format_for_output_table(
                    total_next_x_years_subsidence * area,
                    total_next_x_years_uncertainty * area,
                    current_unit,
                    f"Mm/{next_x_years}yr",
                ),
            }
        )

        # For the shapefile, we want to keep the volume numbers as floats.
        shape_rows.append(
            {
                "Gebied": gebied,
                "Oppervlakte": area,
                f"Total_Volume_last_{last_x_years}_Mm3": total_last_x_years_subsidence
                * area
                * units.calculate_dzdt_factor(current_unit, f"Mm/{last_x_years}yr"),
                f"Total_Volume_next_{next_x_years}_Mm3": total_next_x_years_subsidence
                * area
                * units.calculate_dzdt_factor(current_unit, f"Mm/{next_x_years}yr"),
                f"Total_Volume_last_{last_x_years}_unc": total_last_x_years_uncertainty
                * area
                * units.calculate_dzdt_factor(current_unit, f"Mm/{last_x_years}yr"),
                f"Total_Volume_next_{next_x_years}_unc": total_next_x_years_uncertainty
                * area
                * units.calculate_dzdt_factor(current_unit, f"Mm/{next_x_years}yr"),
            }
        )

    # Create dataframes
    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(
        shape_rows, geometry=subsidence_areas.geometry, crs=subsidence_areas.crs
    )

    # Save to Excel
    logger.info("Saving zonal statistics and volumes to output Excel...")
    df.to_excel(
        Path(config["output_paths"]["base"]) / config["output_paths"]["output_excel"],
        index=False,
    )

    # Save results to a shapefile
    logger.info("Saving zonal statistics and volumes to output shapefile...")
    out_shape = (
        Path(config["output_paths"]["base"]) / config["output_paths"]["output_shape"]
    )
    gdf.to_file(
        out_shape.with_suffix(".gpkg"),
        layer="subsidence_volumes",
        driver="GPKG",
    )

    logger.info("Done!")
