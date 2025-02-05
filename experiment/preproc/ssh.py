import xarray as xr


DEFAULT_STEPS = (
    "lon_to_180_180",
    "restrict_vars"
    "time_lat_lon"
)


def lon_to_180_180(ds: xr.Dataset) -> xr.Dataset:
    ds["longitude"] = (ds.longitude + 180) % 360 - 180
    return ds.sortby(ds.longitude)


def restrict_vars(ds: xr.Dataset) -> xr.Dataset:
    return ds[["adt", "sla"]]


def time_lat_lon(ds):
    return ds.transpose("time", "latitude", "longitude")
