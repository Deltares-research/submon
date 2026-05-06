from functools import singledispatch
from pathlib import Path, WindowsPath

import xarray as xr
from dask.diagnostics import ProgressBar


@singledispatch
def to_nc(
    data: xr.DataArray | xr.Dataset,
    file: str | WindowsPath,
    compress=False,
    compress_level=9,
) -> None:
    """
    Export an xarray.DataArray to a (compressed) NetCDF file using zlib
    compression at the specified level.

    Parameters
    ----------
    data : xr.DataArray | AlignedRasters | TiledRasters
        An xarray.DataArray to save to NetCDF. Can also be an AlignedRasters or
        TiledRasters object.
    file : str | WindowsPath
        Path of the file to be saved.
    compress : bool, optional
        Whether to compress the output file, by default False.
    compress_level : int, optional
        Compression level to use between 1 and 9, by default 9.

    Raises
    ------
    TypeError
        If the input data is not an xarray.DataArray, AlignedRasters, or TiledRasters.
    """
    raise TypeError(f"Unsupported data type: {type(data)}")


@to_nc.register
def _(
    data: xr.DataArray,
    file: str | WindowsPath,
    compress=False,
    compress_level=9,
) -> None:
    """
    Implementation of to_nc for DataArrays.
    """
    if compress:
        delayed = data.to_netcdf(
            file,
            engine="h5netcdf",
            encoding={
                data.name or "__xarray_dataarray_variable__": {
                    "zlib": True,
                    "complevel": compress_level,
                }
            },
            compute=False,
        )

    else:
        delayed = data.to_netcdf(file, engine="h5netcdf", compute=False)

    with ProgressBar():
        print("Exporting to NetCDF...")
        delayed.compute()


@to_nc.register
def _(
    data: xr.Dataset,
    file: str | WindowsPath,
    compress=False,
    compress_level=9,
) -> None:
    """
    Implementation of to_nc for Datasets. Only compresses data_vars with dimensions
    (time, y, x).
    """
    # Define compression parameters for data_vars with dimensions (time, y, x)
    data_vars_compression_parameters = {
        var: {"zlib": True, "complevel": compress_level}
        for var in data.data_vars
        if set(data[var].dims) == {"time", "y", "x"}
    }

    if compress:
        delayed = data.to_netcdf(
            file,
            engine="h5netcdf",
            encoding=data_vars_compression_parameters,
            compute=False,
        )
    else:
        delayed = data.to_netcdf(file, engine="h5netcdf", compute=False)

    with ProgressBar():
        print("Exporting to NetCDF...")
        delayed.compute()


@singledispatch
def to_geotiff(
    data: xr.DataArray | xr.Dataset,
    file: str | WindowsPath,
    compress: bool = False,
    data_var: str | list[str] = None,
) -> None:
    """
    Export an xarray.DataArray with (y, x) dimensions to a compressed GeoTIFF file
    using LZW compression. Also works with an xarray.Dataset (provided which data_var
    to export) and TiledRasters objects.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset | TiledRasters
        The DataArray to save. Must have only two dimensions 'y' and 'x'. If data is an
        xr.Dataset, you must specify which data variable to save.
    file : str | WindowsPath
        Path of the file to be saved.
    compress : bool, optional
        Whether to compress the output file, by default False.
    data_var : str, optional
        The name of the data variable to save if data is an xr.Dataset, by default None.

    Raises
    ------
    TypeError
        If the input data is not an xarray.DataArray, Dataset or TiledRasters object.
    """
    raise TypeError(f"Unsupported data type: {type(data)}")


@to_geotiff.register
def _(
    data: xr.DataArray,
    file: str | WindowsPath,
    compress: bool = False,
    prefix: str = "",
) -> None:
    """
    Implementation of to_geotiff for xr.DataArray objects.
    """

    if file.is_dir():
        name = data.name or "data"
        file = file / f"{prefix}{name}.tif"

    if compress:
        data.rio.to_raster(file, driver="GTiff", compress="LZW")
    else:
        data.rio.to_raster(file, driver="GTiff")


@to_geotiff.register
def _(
    data: xr.Dataset,
    file: str | WindowsPath,
    compress: bool = False,
    prefix: str = "",
) -> None:
    """
    Export all data_vars in an xarray.Dataset to separate GeoTIFF files.
    Each data_var must have dimensions ('y', 'x').
    """
    file = Path(file)

    if not file.is_dir():
        raise ValueError("When exporting an xr.Dataset, 'file' must be a directory.")

    file.mkdir(parents=True, exist_ok=True)

    for var in data.data_vars:
        da = data[var]

        if set(da.dims) != {"y", "x"}:
            raise ValueError(
                f"Cannot export '{var}' to GeoTIFF, dims are {da.dims}. "
                "Select a 2D slice before exporting."
            )

        to_geotiff(
            da,
            file / f"{prefix}{var}.tif",
            compress=compress,
        )
