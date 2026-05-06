import numpy as np
import xarray as xr

from submon.io.export import to_geotiff

STAT_TO_FUNC = {
    "mean": np.nanmean,
    "min": np.nanmin,
    "max": np.nanmax,
    "median": np.nanmedian,
}


def statistics_from_dataarrays(
    subsidence_rasters: list[xr.DataArray],
    stats_to_calculate: list[str] = ["mean", "min", "max"],
) -> xr.Dataset:
    """
    Calculate summary statistics across multiple subsidence scenarios.
    ----------
    Parameters
    ----------
    subsidence_rasters : list[xr.DataArray]
        List of DataArray objects, each representing one scenario.
    stats_to_calculate : list of str, optional
        Statistics to calculate. Allowed values must be keys in STAT_TO_FUNC

    Returns
    -------
    xr.Dataset
        Dataset containing one data variable per requested statistic,
        computed over the 'scenario' dimension.

    """
    scenario_data = xr.concat(subsidence_rasters, dim="scenario")

    results = {}
    for stat in stats_to_calculate:
        if stat not in STAT_TO_FUNC.keys():
            raise ValueError(
                f"Invalid statistic '{stat}' specified. Must be one of {STAT_TO_FUNC.keys()}."
            )

        da = scenario_data.reduce(STAT_TO_FUNC[stat], dim="scenario")

        da.attrs["statistic_type"] = stat

        results[stat] = da

    return xr.Dataset(results)


def predefined_statistics(da: xr.DataArray, factors: dict[str, float]) -> xr.Dataset:
    """
    Calculate predefined statistics (mean, min, max) from a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray containing subsidence values.

    Returns
    -------
    xr.Dataset
        Dataset containing data variables "mean", "min", and "max" computed from the input DataArray.

    """
    if not factors:
        raise ValueError("`factors` must contain at least one statistic")

    return xr.Dataset({name: da * factor for name, factor in factors.items()})


def mean_value_of_dataset_vars(ds: xr.Dataset, skipna: bool = True) -> float:
    """
    Compute mean value for each data variable in a Dataset.
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with multiple data variables.
    skipna : bool, optional
        Whether to skip NaN values, by default True.
    Returns
    -------
    dict[str, float]
        Dictionary with variable names as keys and mean values as values.
    """
    result: dict[str, float] = {}
    for name, da in ds.data_vars.items():
        result[name] = float(da.mean(skipna=skipna))
    return result


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
