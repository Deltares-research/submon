import numpy as np
import xarray as xr

STAT_TO_FUNC = {
    "mean": np.nanmean,
    "min": np.nanmin,
    "max": np.nanmax,
    "median": np.nanmedian,
}


def statistics_from_dataarrays(
    data_arrays: list[xr.DataArray],
    stats_to_calculate: list[str] = ["mean", "min", "max"],
    invert_min_max: bool = False,
) -> xr.Dataset:
    """
    Calculate summary statistics across multiple subsidence scenarios.
    ----------
    Parameters
    ----------
    data_arrays : list[xr.DataArray]
        List of DataArray objects, each representing one scenario.
    stats_to_calculate : list of str, optional
        Statistics to calculate. Allowed values must be keys in STAT_TO_FUNC
    invert_min_max : bool, optional
        Whether to invert the sign of the input data arrays before calculating statistics.
        This can be useful if the input data represents subsidence as negative values and
        you expect that 'maximum subsidence' should correspond to the minimum (most negative)
        value.

    Returns
    -------
    xr.Dataset
        Dataset containing one data variable per requested statistic,
        computed over the 'scenario' dimension.

    """
    scenario_data = xr.concat(data_arrays, dim="scenario")

    results = {}
    for stat in stats_to_calculate:
        if stat not in STAT_TO_FUNC.keys():
            raise ValueError(
                f"Invalid statistic '{stat}' specified. Must be one of {STAT_TO_FUNC.keys()}."
            )

        if invert_min_max:
            if stat == "min":
                func = STAT_TO_FUNC["max"]  # minst daling
            elif stat == "max":
                func = STAT_TO_FUNC["min"]  # meest daling
            else:
                func = STAT_TO_FUNC[stat]
        else:
            func = STAT_TO_FUNC[stat]

        da = scenario_data.reduce(func, dim="scenario")

        da.attrs["statistic_type"] = stat
        results[stat] = da

    return xr.Dataset(results)


def calculate_uncertainty(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate uncertainty measure from 'min' and 'max' data variables in a Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset containing 'min' and 'max' data variables.

    Returns
    -------
    xr.Dataset
        The input Dataset with an 'uncertainty' attribute added, representing the mean
        absolute difference between max and min values divided by 2.

    Raises
    ------
    ValueError
        If the input Dataset does not contain both 'min' and 'max' data variables.

    """
    if "min" not in ds.data_vars and "max" not in ds.data_vars:
        raise ValueError(
            "Input Dataset must contain a 'min' and 'max' data variable to calculate uncertainty."
        )

    uncertainty = np.nanmean(np.abs((ds["max"] - ds["min"]) / 2))

    return uncertainty


def combine_uncertainties(*uncertainties: float) -> float:
    """
    Combine uncertainty measures using the square root of the sum of squares method.

    Parameters
    ----------
    *uncertainties : float
        Variable number of uncertainty measures to combine.

    Returns
    -------
    float
        The combined uncertainty measure.

    """
    return np.sqrt(sum(u**2 for u in uncertainties))


def predefined_statistics(da: xr.DataArray, factors: dict[str, float]) -> xr.Dataset:
    """
    Calculate predefined statistics (mean, min, max) from a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray containing subsidence values.
    factors : dict[str, float]
        Dictionary specifying the factors to apply to the input DataArray for each statistic.
        For example, {"mean": 1.0, "min": 0.75, "max": 1.25} would return a Dataset with
        "mean" equal to the input DataArray, "min" equal to 75% of the input, and "max"
        equal to 125% of the input.

    Returns
    -------
    xr.Dataset
        Dataset containing data variables "mean", "min", and "max" computed from the input DataArray.

    """
    if not all(
        isinstance(k, str) and isinstance(v, (float, int)) for k, v in factors.items()
    ):
        raise TypeError("`factors` must have string keys and float or int values")

    return xr.Dataset({name: da * factor for name, factor in factors.items()})


def mean_value_of_dataset_vars(ds: xr.Dataset, skipna: bool = True) -> dict[str, float]:
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
    result = {}
    for name, da in ds.data_vars.items():
        result[name] = float(da.mean(skipna=skipna))
    return result
