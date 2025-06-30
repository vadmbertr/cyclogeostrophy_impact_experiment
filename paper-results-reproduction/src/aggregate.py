import xarray as xr


def max_of_n_time_rolling_mean(da: xr.DataArray, n_time: int = 7) -> xr.DataArray:
    return da.rolling(time=n_time).mean().max(dim="time")
