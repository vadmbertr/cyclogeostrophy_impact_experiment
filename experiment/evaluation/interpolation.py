from typing import Literal

from jax import vmap
from jaxparrow.tools import operators as jpw_operators
from jaxtyping import Array, Float
import numpy as np
import pyinterp
import pyinterp.backends.xarray
import xarray as xr


def interpolate_grid(
    field: Float[Array, "time lat lon"],
    mask: Float[Array, "time lat lon"],
    axis: Literal[0, 1],
    padding: Literal["left", "right"]
) -> np.ndarray:
    interpolation_map = vmap(jpw_operators.interpolation, in_axes=(0, 0, None, None))
    return np.array(interpolation_map(field, mask, axis, padding), dtype=np.float32)


def interpolate_drifters_location(
    drifter_ds: xr.Dataset,
    time: Float[Array, "time"],
    latitude_v: Float[Array, "lat lon"],
    longitude_u: Float[Array, "lat lon"],
    uv_fields_hat: dict
) -> xr.Dataset:
    y_axis = pyinterp.Axis(latitude_v[:, 0])
    x_axis = pyinterp.Axis(longitude_u[0, :] + 180, is_circle=True)
    t_axis = pyinterp.TemporalAxis(time)

    for method, (u_field, v_field) in uv_fields_hat.items():
        u_grid = pyinterp.Grid3D(x_axis, y_axis, t_axis, u_field.T)
        v_grid = pyinterp.Grid3D(x_axis, y_axis, t_axis, v_field.T)

        drifter_ds[f"u_hat_{method}"] = (
            "obs", pyinterp.trivariate(u_grid, drifter_ds.lon + 180, drifter_ds.lat, drifter_ds.time)
        )
        drifter_ds[f"v_hat_{method}"] = (
            "obs", pyinterp.trivariate(v_grid, drifter_ds.lon + 180, drifter_ds.lat, drifter_ds.time)
        )

    return drifter_ds.drop_vars(["time"])
