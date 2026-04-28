from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from matplotlib.pylab import compress, mean

from submon import stats
from submon.io.export import to_geotiff
from submon.rasters import SubsidenceRaster

STAT_TO_FUNC = {
    "mean": np.nanmean,
    "min": np.nanmin,
    "max": np.nanmax,
    "median": np.nanmedian,
}


def statistics_from_subsidence_rasters(
    subsidence_rasters: list[SubsidenceRaster],
    stats_to_calculate: list[str] = ["mean", "min", "max"],
) -> xr.Dataset:
    """
    Calculate a statistic from a list of SubsidenceRasters.
    """
    scenario_data = xr.concat(
        [raster.da for raster in subsidence_rasters], dim="scenario"
    )

    results = {}
    for stat in stats_to_calculate:
        if stat not in STAT_TO_FUNC.keys():
            raise ValueError(
                f"Invalid statistic '{stat}' specified. Must be one of {STAT_TO_FUNC.keys()}."
            )

        results[stat] = scenario_data.reduce(STAT_TO_FUNC[stat], dim="scenario")

    return xr.Dataset(
        data_vars=results,
    )


def total_subsidence_with_mining_uncertainty(
    combined: xr.Dataset,
    mining: xr.Dataset,
    unc_last30: float = 0.25,
    unc_next30: float = 0.50,
    period_dim: str = "period",
) -> xr.Dataset:
    """
    Bereken totale bodemdaling = combined (GIA+Tect) + winning (mining),
    waarbij onzekerheid (%) alleen op winning wordt toegepast.

    Parameters
    ----------
    combined : xr.Dataset
        Dataset met data_vars: "min", "mean", "max" (geologisch: GIA+Tect).
    mining : xr.Dataset
        Dataset met 1 variabele (winning kaart).
    unc_last30 : float
        Onzekerheid rondom winning voor laatste 30 jaar (0.25 = ±25%).
    unc_next30 : float
        Onzekerheid rondom winning voor komende 30 jaar (0.50 = ±50%).
    period_dim : str
        Naam van de dimensie voor periode labels.

    Returns
    -------
    xr.Dataset
        Dataset met dims: (period, y, x) en data_vars: min/mean/max (totale bodemdaling).
    """
    # Validate combined vars
    required = {"min", "mean", "max"}
    if not required.issubset(set(combined.data_vars)):
        raise ValueError(
            f"`combined` must contain {required}, got {list(combined.data_vars)}"
        )

    # Extract mining DataArray from Dataset (keep metadata)
    mvars = list(mining.data_vars)
    if len(mvars) != 1:
        raise ValueError(f"`mining` must contain exactly 1 variable, got {mvars}")
    m = mining[mvars[0]]

    # Helper to build total for one uncertainty level
    def _total_for_unc(u: float) -> xr.Dataset:
        return xr.Dataset(
            data_vars={
                "min": combined["min"] + m * (1.0 - u),
                "mean": combined["mean"] + m,
                "max": combined["max"] + m * (1.0 + u),
            },
            coords=combined.coords,
            attrs=dict(combined.attrs),
        )

    total_last30 = _total_for_unc(float(unc_last30))
    total_next30 = _total_for_unc(float(unc_next30))

    # Concat into one Dataset
    out = xr.concat([total_last30, total_next30], dim=period_dim)
    out = out.assign_coords({period_dim: ["last30", "next30"]})

    return out


def export_dataset_vars_to_geotiff(
    ds: xr.Dataset,
    out_dir: Path,
    compress: bool = True,
):
    """
    Schrijf elke data_var in een xr.Dataset weg als losse GeoTIFF.
    Verwacht dat elke data_var 2D is (y, x).
    """
    for var in ds.data_vars:
        out_tif = out_dir / f"{var}.tif"
        to_geotiff(
            ds,
            out_tif,
            compress=compress,
            data_var=var,
        )


def fix_nodata(da, threshold=-1e30):
    """
    Zet extreme NoData waarden om naar NaN en schrijf NoData correct weg.
    """
    da = da.where(da > threshold)
    da = da.rio.write_nodata(np.nan, inplace=False)
    return da


def zonal_stats(da, geom, crs):
    """
    Bereken zonale statistieken van een DataArray binnen geometrieën.
    """

    clipped = da.rio.clip([geom], crs=crs, drop=True, all_touched=True)
    if clipped.isnull().all():
        return (np.nan, np.nan, np.nan)

    return (
        float(clipped.mean(skipna=True)),
        float(clipped.min(skipna=True)),
        float(clipped.max(skipna=True)),
    )


def zonal_bandwidth_from_dataset(ds: xr.Dataset, geom, crs, nd=3):
    """
    zonal_stats plus minus
    ds bevat data_vars: 'mean', 'min', 'max'
    Geeft (mean, min, max, 'mean ±') voor één polygon
    """
    mean, _, _ = zonal_stats(ds["mean"], geom, crs)
    minv, _, _ = zonal_stats(ds["min"], geom, crs)
    maxv, _, _ = zonal_stats(ds["max"], geom, crs)

    txt = f"{mean:.{nd}f} ± {(maxv - minv) / 2:.{nd}f}"
    return mean, minv, maxv, txt


def zonal_mining_with_uncertainty(da, geom, crs, unc, nd=3):
    mean = zonal_mean(da, geom, crs)
    if np.isnan(mean):
        return np.nan, np.nan, np.nan, ""

    minv = mean * (1 - unc)
    maxv = mean * (1 + unc)
    txt = f"{mean:.{nd}f} ± {abs(mean) * unc:.{nd}f}"
    return mean, minv, maxv, txt


def zonal_mean(da, geom, crs):
    clipped = da.rio.clip([geom], crs=crs, drop=True, all_touched=True)
    if clipped.isnull().all():
        return np.nan
    return float(clipped.mean(skipna=True))


def volume(mean_value, area_m2):
    return mean_value * area_m2 / 1e6
