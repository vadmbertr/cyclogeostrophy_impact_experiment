import clouddrift as cd
import numpy as np
from scipy import signal
import xarray as xr


DEFAULT_STEPS = (
    "gps_only",
    "svp_only",
    "drogued_only",
    "remove_low_latitudes",
    "finite_value_only",
    "remove_outlier_values",
    "restrict_vars"
)


def gps_only(ds: xr.Dataset) -> xr.Dataset:
    ds.location_type.load()
    return cd.ragged.subset(ds, {"location_type": True}, row_dim_name="traj")  # True means GPS / False Argos


def svp_only(ds: xr.Dataset) -> xr.Dataset:
    ds.typebuoy.load()
    return cd.ragged.subset(
        ds,
        {"typebuoy": lambda tb: np.char.find(tb.astype(str), "SVP") != -1},
        row_dim_name="traj"
    )


def drogued_only(ds: xr.Dataset) -> xr.Dataset:
    ds.drogue_status.load()
    return cd.ragged.subset(ds, {"drogue_status": True}, row_dim_name="traj")


def remove_low_latitudes(ds: xr.Dataset, cutoff: float = 5) -> xr.Dataset:
    ds.lat.load()
    return cd.ragged.subset(ds, {"lat": lambda arr: np.abs(arr) > cutoff}, row_dim_name="traj")


def finite_value_only(ds: xr.Dataset) -> xr.Dataset:
    ds.lon.load()
    ds = cd.ragged.subset(ds, {"lon": np.isfinite}, row_dim_name="traj")
    ds.lat.load()
    ds = cd.ragged.subset(ds, {"lat": np.isfinite}, row_dim_name="traj")
    ds.ve.load()
    ds = cd.ragged.subset(ds, {"ve": np.isfinite}, row_dim_name="traj")
    ds.vn.load()
    ds = cd.ragged.subset(ds, {"vn": np.isfinite}, row_dim_name="traj")
    ds.time.load()
    ds = cd.ragged.subset(ds, {"time": lambda arr: ~np.isnat(arr)}, row_dim_name="traj")
    return ds


def remove_outlier_values(ds: xr.Dataset, cutoff: float = 10) -> xr.Dataset:
    def velocity_cutoff(arr: xr.DataArray) -> xr.DataArray:
        return np.abs(arr) <= cutoff

    ds.ve.load()
    ds = cd.ragged.subset(ds, {"ve": velocity_cutoff}, row_dim_name="traj")
    ds.vn.load()
    ds = cd.ragged.subset(ds, {"vn": velocity_cutoff}, row_dim_name="traj")
    return ds


def restrict_vars(ds: xr.Dataset) -> xr.Dataset:
    return ds[["rowsize", "lon", "lat", "time", "ve", "vn"]]
