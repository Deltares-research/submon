from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray as rio
import xarray as xr
from scipy.spatial import cKDTree

from submon import utils


def read_xyz(
    xyz_file: str | Path,
    resolution: float | int = None,
    gridded: bool = True,
    non_gridded_tolerance: float = 1,
):
    """
    Reads an XYZ file and returns the data as a DataArray.

    Parameters
    ----------
    xyz_file : str
        Path to the XYZ file. The file extension should be one of [".xyz", ".txt", ".csv", ".pts"].
    resolution : float | int, optional
        The resolution for gridding the data if `gridded` is False. This parameter is required if `gridded` is False.
    gridded : bool, optional
        If True, the data will be treated as already gridded points. If False, the data
        will be treated as random points. Default is True.
    non_gridded_tolerance : float, optional
        Tolerance level for non-gridded points. Only used if `gridded` is False. Default
        is 1. This means that the gridding algorith will search for points within the 1
        times resolution distance.

    Returns
    -------
    da : DataArray
        The data from the XYZ file as a DataArray.

    Raises
    ------
    NameError
        If the file extension is not one of [".xyz", ".txt", ".csv", ".pts"].

    Notes
    -----
    The gridded kwarg must be checked by the user. By default it is True and therefore it
    is assumed that the data is already gridded. This makes processing faster, so if the
    data is indeed gridded there is no point in setting gridded to False.

    Tips:
    If you see a lot of missing data, perhaps the data is not gridded. Try setting gridded
    argument to False and adjust the non_gridded_tolerance argument. If you still see missing
    data, try increasing the non_gridded_tolerance argument slightly.
    """
    if Path(xyz_file).suffix in [".xyz", ".txt", ".csv", ".pts"]:
        file = Path(xyz_file)
    else:
        raise NameError(
            "File extension not supported. Supported extensions are '.xyz', '.txt', '.csv', '.pts'."
        )

    # Reading the XYZ-file
    sep = utils.find_xyz_sep(xyz_file)
    if sep == " ":
        df = pd.read_csv(file, header=None, names=["x", "y", "z"], sep="\s+")
    else:
        df = pd.read_csv(file, sep=sep, header=None, names=["x", "y", "z"])

    data = utils.remove_df_header(df)
    if gridded:
        da = __xyz_gridded_points(data)
    else:
        da = __xyz_random_points(data, resolution, non_gridded_tolerance)

    return da


def __xyz_gridded_points(data: pd.DataFrame) -> xr.DataArray:
    """
    Convert XYZ point data that already represents a grid into xarray DataArray.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame containing the XYZ point data with columns 'x', 'y', and 'z'.

    Returns
    -------
    xr.DataArray
        A gridded DataArray with the Z-values.

    Notes
    -----
    The function sorts the data by 'y' in descending order and 'x' in ascending order,
    calculates the grid indices, and fills the grid with the Z-values.
    """
    bounds = (
        np.nanmin(data.x),
        np.nanmax(data.x),
        np.nanmin(data.y),
        np.nanmax(data.y),
    )

    sorted_data = data.sort_values(["y", "x"], ascending=[True, True])

    coord_x = np.unique(sorted_data["x"])
    coord_y = np.unique(sorted_data["y"])

    res_x = np.mean(np.diff(coord_x))
    res_y = np.mean(np.diff(coord_y))

    idxs_x = np.int32(np.round((sorted_data["x"] - bounds[0]) / res_x))
    idxs_y = np.int32(np.round((bounds[-1] - sorted_data["y"]) / res_y))

    # DataArray Attributes
    transform_attr = (
        res_x,
        0.0,
        coord_x[0] - 0.5 * res_x,
        0.0,
        -res_y,
        coord_y[0] + 0.5 * res_y,
    )
    res_attr = (res_x, res_y)
    is_tiled_attr = 0
    scales_attr = (1.0,)
    offset_attr = (0.0,)

    # Allocate array
    array = np.zeros((len(coord_y), len(coord_x)), dtype=np.float32)

    # Fill array
    array[idxs_y, idxs_x] = sorted_data["z"].values
    array[array == 0] = np.nan
    array = array[::-1, :]
    da = xr.DataArray(data=array, coords={"y": coord_y, "x": coord_x}, dims=["y", "x"])
    da = utils.set_da_attributes(
        da,
        transform=transform_attr,
        res=res_attr,
        is_tiled=is_tiled_attr,
        scales=scales_attr,
        offset=offset_attr,
        nodatavals=tuple([np.nan]),
    )
    return da


def __xyz_random_points(
    data,
    res,
    non_gridded_tolerance,
):
    """
    Generate a gridded DataArray from random XYZ points.

    Parameters
    ----------
    data : pandas.DataFrame
        A DataFrame containing 'x', 'y', and 'z' columns representing the coordinates and
        of values.
    res : float
        The desired resolution of the output grid.
    non_gridded_tolerance : float
        The tolerance factor for considering points within the resolution.

    Returns
    -------
    xarray.DataArray
        A DataArray with the gridded 'z' values, and coordinates 'x' and 'y'.

    Notes
    -----
    This function creates a grid of points based on the provided resolution and uses a KDTree
    for fast spatial queries to find the nearest points within the specified tolerance. The nearest
    'z' values of the valid points are computed and reshaped into the grid.
    """
    bounds = (
        np.nanmin(data.x),
        np.nanmax(data.x),
        np.nanmin(data.y),
        np.nanmax(data.y),
    )

    coord_x = np.arange(bounds[0], bounds[1] + res, res, dtype=np.float32)
    coord_y = np.arange(bounds[3], bounds[2] - res, -res, dtype=np.float32)

    array = np.full((len(coord_y), len(coord_x)), np.nan, dtype=np.float32)

    # Create a KDTree for fast spatial queries
    tree = cKDTree(data[["x", "y"]].values)

    # Generate grid points
    grid_x, grid_y = np.meshgrid(coord_x, coord_y)
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

    # Query the tree for points within the resolution
    distances, indices = tree.query(
        grid_points, distance_upper_bound=res * non_gridded_tolerance
    )

    # Filter out points that are beyond the resolution
    valid_mask = distances <= res * non_gridded_tolerance
    valid_indices = indices[valid_mask]

    # Compute nearest z-values for valid points
    mean_z_values = np.full(grid_points.shape[0], np.nan, dtype=np.float32)
    mean_z_values[valid_mask] = data["z"].values[valid_indices]

    # Reshape the result back into the grid shape
    array = mean_z_values.reshape(grid_x.shape)

    da = xr.DataArray(data=array, coords={"y": coord_y, "x": coord_x}, dims=["y", "x"])
    da = utils.set_da_attributes(
        da,
        transform=(res, 0.0, coord_x[0] - 0.5 * res, 0.0, -res, coord_y[0] + 0.5 * res),
        res=(res, res),
        is_tiled=0,
        scales=(1.0,),
        offset=(0.0,),
        nodatavals=(np.nan,),
    )
    return da
