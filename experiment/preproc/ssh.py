import xarray as xr


DEFAULT_STEPS = (
    "lon_to_180_180",
    "time_lat_lon"
)


def lon_to_180_180(ds: xr.Dataset) -> xr.Dataset:
    ds["longitude"] = (ds.longitude + 180) % 360 - 180
    ds = ds.sortby(ds.longitude)
    return ds


def time_lat_lon(ds):
    return ds.transpose("time", "latitude", "longitude")
