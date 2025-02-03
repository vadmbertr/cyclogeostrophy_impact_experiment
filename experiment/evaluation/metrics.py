from typing import List, Tuple

import numpy as np
import pandas as pd
import pyinterp
import xarray as xr


def _euclidean_dist(
    u: xr.DataArray, u_hat: xr.DataArray,
    v: xr.DataArray, v_hat: xr.DataArray
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    err_u = (u_hat - u)
    err_v = (v_hat - v)
    err_uv = (err_u ** 2 + err_v ** 2) ** (1/2)
    return err_u, err_v, err_uv


def compute_along_traj_metrics(drifter_ds: xr.Dataset, methods: List[str]) -> pd.DataFrame:
    traj_metrics_ds = drifter_ds
    for method in methods:
        traj_metrics_ds[f"err_u_{method}"], traj_metrics_ds[f"err_v_{method}"], traj_metrics_ds[f"err_{method}"] = (
            _euclidean_dist(drifter_ds.ve, drifter_ds[f"u_hat_{method}"], drifter_ds.vn, drifter_ds[f"v_hat_{method}"])
        )
        traj_metrics_ds = traj_metrics_ds.drop_vars([f"u_hat_{method}", f"v_hat_{method}"])

    traj_metrics_ds = traj_metrics_ds.drop_vars(["ve", "vn"])
    traj_metrics_df = traj_metrics_ds.to_dataframe()

    traj_metrics_df.attrs = traj_metrics_ds.attrs.copy()
    for var in traj_metrics_ds.variables:
        traj_metrics_df[var].attrs = traj_metrics_ds[var].attrs.copy()

    return traj_metrics_df


def compute_binned_metrics(
    field_ds: xr.Dataset,
    traj_metrics: xr.Dataset,
    methods: List[str],
    bin_size: int
) -> Tuple[xr.Dataset, xr.Dataset]:
    latitude = np.arange(field_ds.latitude.min(), field_ds.latitude.max(), bin_size) + bin_size / 2
    longitude = np.arange(field_ds.longitude.min(), field_ds.longitude.max(), bin_size) + bin_size / 2

    binning = pyinterp.Binning2D(pyinterp.Axis(longitude + 180, is_circle=True), pyinterp.Axis(latitude))

    what_attrs = {
        "err_u": "$\\langle \\epsilon_{u} \\rangle$",
        "err_v": "$\\langle \\epsilon_{v} \\rangle$",
        "err": "$\\langle \\epsilon \\rangle$",
    }

    data_vars_sum = {}
    data_vars_count = {}
    for method in methods:
        for expr_prefix in ["err_u", "err_v", "err"]:
            expr = f"{expr_prefix}_{method}"
            binning.clear()
            binning.push(traj_metrics.lon + 180, traj_metrics.lat, traj_metrics[expr], False)  # noqa
            data_vars_sum[f"{expr}"] = (
                ["latitude", "longitude"],
                binning.variable("sum").T,
                {"method": method, "what": what_attrs[expr_prefix], "units": "$m/s$"}
            )
            data_vars_count[f"{expr}"] = (["latitude", "longitude"], binning.variable("sum_of_weights").T)

    uv_coords = {
        "latitude": (["latitude"], latitude.astype(np.float32)),
        "longitude": (["longitude"], longitude.astype(np.float32))
    }
    errors_sum_ds = xr.Dataset(data_vars=data_vars_sum, coords=uv_coords)
    errors_count_ds = xr.Dataset(data_vars=data_vars_count, coords=uv_coords)

    return errors_sum_ds, errors_count_ds
