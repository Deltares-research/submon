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
    Calculate summary statistics across multiple subsidence scenarios.
    ----------
    Parameters
    ----------
    subsidence_rasters : list[SubsidenceRaster]
        List of SubsidenceRaster objects, each representing one scenario.
    stats_to_calculate : list of str, optional
        Statistics to calculate. Allowed values must be keys in STAT_TO_FUNC
        (e.g. ["mean", "min", "max"]).

    Returns
    -------
    xr.Dataset
        Dataset containing one data variable per requested statistic,
        computed over the 'scenario' dimension.

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

    Compute total subsidence = combined (GIA + Tectonics) + mining,
    where uncertainty (%) is applied to the mining component only.

    Parameters
    ----------
    combined : xr.Dataset
        Dataset containing data variables "min", "mean", and "max"
        representing geological subsidence (GIA + tectonics).
    mining : xr.Dataset
        Dataset containing a single variable representing mining-induced subsidence.
    unc_last30 : float
        Relative uncertainty for mining during the past 30 years
        (0.25 = ±25%).
    unc_next30 : float
        Relative uncertainty for mining during the next 30 years
        (0.50 = ±50%).
    period_dim : str
        Name of the dimension used for period labels.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions (period, y, x) and data variables
        "min", "mean", and "max" representing total subsidence.

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
    """
    for var in ds.data_vars:
        out_tif = out_dir / f"{var}.tif"
        to_geotiff(
            ds,
            out_tif,
            compress=compress,
            data_var=var,
        )

def zonal_stats(da, geom, crs):
    """

    Compute zonal statistics of a DataArray within a geometry.

    Parameters
    ----------
    da : xr.DataArray
        Input raster data.
    geom : shapely geometry
        Geometry defining the zone over which statistics are calculated.
    crs : CRS
        Coordinate reference system of the geometry.

    Returns
    -------
    Float
        (mean, min, max) values within the geometry.
        Returns (np.nan, np.nan, np.nan) if no valid data is present.

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

    Compute zonal statistics with a plus/minus bandwidth from a dataset.

    The input dataset must contain the data variables "mean", "min", and "max".
    For a single polygon, this function returns the zonal mean, minimum,
    maximum, and a formatted string representing:
    mean ± half the range ((max - min) / 2).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing data variables "mean", "min", and "max".
    geom : shapely geometry
        Polygon geometry over which zonal statistics are calculated.
    crs : CRS
        Coordinate reference system of the geometry.
    nd : int, optional
        Number of decimal places used in the formatted output string.

    Returns
    -------
    tuple
        (mean, min, max, txt), where txt is formatted as "mean ± bandwidth".

    """
    mean, _, _ = zonal_stats(ds["mean"], geom, crs)
    minv, _, _ = zonal_stats(ds["min"], geom, crs)
    maxv, _, _ = zonal_stats(ds["max"], geom, crs)

    txt = f"{mean:.{nd}f} ± {(maxv - minv) / 2:.{nd}f}"
    return mean, minv, maxv, txt


def zonal_mining_with_uncertainty(da, geom, crs, unc, nd=3):
    """

    Compute the zonal mean value and associated uncertainty band
    based on a relative uncertainty.

    Parameters
    ----------
    da : xr.DataArray
        DataArray containing mining-related values.
    geom : shapely geometry
        Geometry defining the area of interest.
    crs : CRS
        Coordinate reference system of the geometry.
    unc : float
        Relative uncertainty (e.g. 0.25 for ±25%).
    nd : int, optional
        Number of decimal places used in the formatted output.

    Returns
    -------
    mean : float
        Zonal mean value.
    minv : float
        Lower bound (mean * (1 - unc)).
    maxv : float
        Upper bound (mean * (1 + unc)).
    txt : str
        Formatted string: "mean ± absolute uncertainty".


    """
    mean = zonal_mean(da, geom, crs)
    if np.isnan(mean):
        return np.nan, np.nan, np.nan, ""

    minv = mean * (1 - unc)
    maxv = mean * (1 + unc)
    txt = f"{mean:.{nd}f} ± {abs(mean) * unc:.{nd}f}"
    return mean, minv, maxv, txt


def zonal_mean(da, geom, crs):
    """

    Compute the zonal mean value within a geometry.

    Parameters
    ----------
    da : xr.DataArray
        Raster data.
    geom : shapely geometry
        Geometry defining the area of interest.
    crs : CRS
        Coordinate reference system of the geometry.

    Returns
    -------
    float
        Mean value within the geometry, or np.nan if no valid data is present.

    """

    clipped = da.rio.clip([geom], crs=crs, drop=True, all_touched=True)
    if clipped.isnull().all():
        return np.nan
    return float(clipped.mean(skipna=True))


def volume(mean_value, area_m2):
    """
    Convert a mean value and area to a volume in millions.

    Parameters
    ----------
    mean_value : float
        Mean value (e.g. subsidence in metres).
    area_m2 : float
        Area in square metres.

    Returns
    -------
    float
        Volume expressed in millions (division by 1e6).
    """

    return mean_value * area_m2 / 1e6
