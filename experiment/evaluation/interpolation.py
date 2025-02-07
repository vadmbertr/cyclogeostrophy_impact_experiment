from typing import Dict, Literal, Tuple

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
    latitude_u: Float[Array, "lat lon"],
    longitude_u: Float[Array, "lat lon"],
    latitude_v: Float[Array, "lat lon"],
    longitude_v: Float[Array, "lat lon"],
    uv_fields_hat: Dict[str, Tuple[np.ndarray, np.ndarray]],
    previous_uv_fields: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> xr.Dataset:
    if previous_uv_fields is not None:
        time = np.concat([time[0:1] - (time[1] - time[0]), time])
        
    u_y_axis = pyinterp.Axis(latitude_u[:, 0])
    u_x_axis = pyinterp.Axis(longitude_u[0, :] + 180, is_circle=True)
    v_y_axis = pyinterp.Axis(latitude_v[:, 0])
    v_x_axis = pyinterp.Axis(longitude_v[0, :] + 180, is_circle=True)
    t_axis = pyinterp.TemporalAxis(time)

    for method, (u_field, v_field) in uv_fields_hat.items():
        if previous_uv_fields is not None:
            previous_u_field, previous_v_field = previous_uv_fields[method]
            u_field = np.concat([previous_u_field, u_field])
            v_field = np.concat([previous_v_field, v_field])
        
        u_grid = pyinterp.Grid3D(u_x_axis, u_y_axis, t_axis, u_field.T)
        v_grid = pyinterp.Grid3D(v_x_axis, v_y_axis, t_axis, v_field.T)

        drifter_ds[f"u_hat_{method}"] = (
            "obs", pyinterp.trivariate(u_grid, drifter_ds.lon + 180, drifter_ds.lat, drifter_ds.time)
        )
        drifter_ds[f"v_hat_{method}"] = (
            "obs", pyinterp.trivariate(v_grid, drifter_ds.lon + 180, drifter_ds.lat, drifter_ds.time)
        )

    return drifter_ds
