import tomllib
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from submon import rasters, stats
from submon.io import export, read
from submon.io.export import to_geotiff, to_nc

""" Compute subsidence components (GIA, Tectonics, Combined, Mining) and total subsidence with mining uncertainty. Export results to NetCDF and GeoTIFF, and calculate zonal statistics for specified areas. """

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
    gia_stats = stats.statistics_from_subsidence_rasters(
        data["gia"], config["gia"]["stats"]
    )

    # Tectonic subsidence statistics
    tect_stats = xr.Dataset(
        data_vars={x.statistic_type: x.da for x in data["tectonic"]}
    )

    for var in tect_stats.data_vars:
        tect_stats[var] = stats.fix_nodata(tect_stats[var])

    # Combined (GIA + Tectonics)
    combined_stats = rasters.sum_subsidence_rasters(gia_stats, tect_stats)

    # mining data
    mining_stats = xr.Dataset(data_vars={"mining": data["mining"][0].da})
    mining_stats["mining"] = stats.fix_nodata(mining_stats["mining"])

    # Total subsidence with mining uncertainty
    # last30 / next30 represent uncertainty scenarios
    total = stats.total_subsidence_with_mining_uncertainty(
        combined_stats,
        mining_stats,
        unc_last30=0.25,
        unc_next30=0.50,
    )
    # Export all subsidence components to NetCDF for reuse and reproducibility
    # (one file per component)

    exports = [
        ("gia", gia_stats, "gia.nc"),
        ("tectonic", tect_stats, "tect.nc"),
        ("combined", combined_stats, "subsidence_combined.nc"),
        ("mining", mining_stats, "mining.nc"),
        ("total", total, "subsidence_total.nc"),
    ]

    for key, ds, filename in exports:
        to_nc(
            ds,
            Path(config["output"]["base"]) / config["output"]["paths"][key] / filename,
        )

    # Export to GEOTIFF
    # gebruik de helper functie export_dataset_vars_to_geotiff om alle data_vars in een Dataset te exporteren naar aparte GeoTIFFs (to_geotiff kan maar 1 dataarray per keer)
    for key, ds in [
        ("gia", gia_stats),
        ("tectonic", tect_stats),
        ("combined", combined_stats),
        ("mining", mining_stats),
    ]:
        stats.export_dataset_vars_to_geotiff(
            ds,
            Path(config["output"]["base"]) / config["output"]["paths"][key],
        )

    for period in total["period"].values:
        total_p = total.sel(period=period)

        period_dir = (
            Path(config["output"]["base"])
            / config["output"]["paths"]["total"]
            / str(period)
        )
        period_dir.mkdir(parents=True, exist_ok=True)

        for stat in ["min", "mean", "max"]:
            to_geotiff(
                total_p,
                period_dir / f"{stat}.tif",
                compress=True,
                data_var=stat,
            )

    # deelgebieden inlezen
    gdf = gpd.read_file(Path(config["investigated_areas"]["path"]))
    gdf = gdf.to_crs(combined_stats.rio.crs)
    gdf["area_m2"] = gdf.geometry.area

    # Lege lijst, elk deelgebied wordt 1 dictionary
    rows = []
    # loop over de deelgebieden en bereken zonale statistieken voor GIA, Tect, Combined, Mining, Total
    for i, row in gdf.iterrows():
        geom = row.geometry
        gebied = row["Gebied"]
        area = row["area_m2"]

        # GIA:
        gia_mean, gia_min, gia_max, gia_txt = stats.zonal_bandwidth_from_dataset(
            gia_stats, geom, gdf.crs
        )

        # tectonics:
        tect_mean, tect_min, tect_max, tect_txt = stats.zonal_bandwidth_from_dataset(
            tect_stats, geom, gdf.crs
        )
        # combined:
        comb_mean, comb_min, comb_max, comb_txt = stats.zonal_bandwidth_from_dataset(
            combined_stats, geom, gdf.crs
        )

        # mining

        mining_last30_mean, mining_last30_min, mining_last30_max, mining_last30_text = (
            stats.zonal_mining_with_uncertainty(
                mining_stats["mining"], geom, gdf.crs, unc=0.25
            )
        )

        mining_next30_mean, mining_next30_min, mining_next30_max, mining_next30_text = (
            stats.zonal_mining_with_uncertainty(
                mining_stats["mining"], geom, gdf.crs, unc=0.50
            )
        )

        # total
        total_last30_mean, total_last30_min, total_last30_max, total_last30_text = (
            stats.zonal_bandwidth_from_dataset(
                total.sel(period="last30"), geom, gdf.crs
            )
        )
        total_next30_mean, total_next30_min, total_next30_max, total_next30_text = (
            stats.zonal_bandwidth_from_dataset(
                total.sel(period="next30"), geom, gdf.crs
            )
        )

        rows.append(
            {
                "Gebied": gebied,
                # Componenten zonder periode
                "GIA": gia_txt,
                "Tect": tect_txt,
                "CombinedGeology": comb_txt,
                # Mining en total zijn wel periode-afhankelijk
                "Mining_last30": mining_last30_text,
                "Total_last30": total_last30_text,
                "Mining_next30": mining_next30_text,
                "Total_next30": total_next30_text,
                # Volumes
                "Vol_Mm3_GIA": stats.volume(gia_mean, area),
                "Vol_Mm3_Tect": stats.volume(tect_mean, area),
                "Vol_Mm3_CombinedGeology": stats.volume(comb_mean, area),
                "Vol_Mm3_Mining_last30": stats.volume(mining_last30_mean, area),
                "Vol_Mm3_Total_last30": stats.volume(total_last30_mean, area),
                "Vol_Mm3_Total_next30": stats.volume(total_next30_mean, area),
            }
        )
        df = pd.DataFrame(rows)
        df.to_excel(
            Path(config["output"]["base"]) / config["output"]["files"]["excel"],
            index=False,
        )

    print("Done!")
