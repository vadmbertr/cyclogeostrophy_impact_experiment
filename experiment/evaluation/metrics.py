from typing import List, Tuple

import pandas as pd
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
 