import numpy as np
import pandas as pd
import xarray as xr


def remove_equatorial_band(ds: xr.Dataset) -> xr.Dataset:
    return ds.where(np.abs(ds.latitude) > 5)


def restrict_to_gulfstream(ds: xr.Dataset) -> xr.Dataset:
    return ds.sel(latitude=slice(33.5, 42.5), longitude=slice(-73.5, -52.5))


def restrict_to_mediterranean_sea(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.sel(latitude=slice(30.2639, 45.7833), longitude=slice(-6.0327, 36.2173))
    return ds.where(
        np.logical_not(
            ((ds.latitude >= 43.2744) & (ds.longitude <= -.1462)) |  # Black Sea
            ((ds.latitude >= 40.9088)  & (ds.longitude >= 27.4437))  # Bay of Biscay
        )
    )


def restrict_df_to_neurost_timeperiod(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["time"] >= np.datetime64("2010")) & (df["time"] < np.datetime64("2023"))]


def restrict_ds_to_neurost_timeperiod(ds: xr.Dataset) -> xr.Dataset:
    return ds.sel(time=slice("2010", "2023"))
