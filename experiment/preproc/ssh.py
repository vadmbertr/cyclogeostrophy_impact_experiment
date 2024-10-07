import xarray as xr


def lon_to_180_180(ds: xr.Dataset) -> xr.Dataset:
    ds["longitude"] = (ds["longitude"] + 180) % 360 - 180
    return ds
