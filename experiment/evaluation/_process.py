from collections.abc import Callable
import math
import os
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
from jaxparrow.tools.geometry import compute_coriolis_factor, compute_spatial_step
from jaxtyping import Array, Float
import xarray as xr

from ..io.experiment import ExperimentData
from ..io.ssh import SSHData
from ..logger._logger import LOGGER
from .loss import compute_loss_value_and_grad


def _estimate_batch_indices(n_time: int, n_lat: int, n_lon: int, memory_per_device: float) -> List[int]:
    f32_size = 4  # mostly manipulate f32 arrays
    comp_mem_per_time = f32_size * n_lat * n_lon * 1e-9  # in Gb
    comp_mem_per_time *= 40  # empirical factor preventing OOM errors
    batch_size = int(memory_per_device // comp_mem_per_time)  # conservative batch size
    n_batches = math.ceil(n_time / batch_size)  # conservative number of batches
    indices = jnp.arange(1, n_batches) * batch_size
    return indices.tolist()


def estimate_and_evaluate(
    experiment_data: ExperimentData,
    experiment_config: str,
    ssh_data: SSHData,
    cyclogeostrophy_fun: Callable,
    memory_per_device: int
) -> Tuple[xr.Dataset, xr.Dataset]:
    ssh_ds = ssh_data.dataset

    # fix over batches
    lon_t, lat_t = jnp.meshgrid(ssh_ds.longitude.values, ssh_ds.latitude.values)

    # estimate batch indices based on domain dimensions
    n_time, n_lat, n_lon = tuple(ssh_ds.sizes.values())
    batch_indices = _estimate_batch_indices(n_time, n_lat, n_lon, memory_per_device)
    batch_indices = [0] + batch_indices + [n_time]

    # apply per batch
    for idx0, idx1 in zip(batch_indices[:-1], batch_indices[1:]):
        LOGGER.debug(f"2.i.0. mini-batch {[idx0, idx1]}")
        process_batch(
            idx0, idx1,
            ssh_ds,
            cyclogeostrophy_fun,
            lat_t, lon_t,
            experiment_data, experiment_config
        )


def _estimate_ssc(
    cyclogeostrophy_fun: Callable,
    ssh_t: Float[Array, "time lat lon"],
    lat_t: Float[Array, "lat lon"],
    lon_t: Float[Array, "lat lon"]
) -> Tuple[
    Dict[str, Tuple[np.ndarray, np.ndarray]],
    Float[Array, "lat lon"],
    Float[Array, "lat lon"],
    Float[Array, "lat lon"],
    Float[Array, "lat lon"],
    Float[Array, "time lat lon"]
]:
    mask = ~np.isfinite(ssh_t)
    u_cyclo, v_cyclo, u_geos, v_geos, lat_u, lon_u, lat_v, lon_v = cyclogeostrophy_fun(lat_t, lon_t, ssh_t, mask)

    # nan incoherent velocities
    u_cyclo = jnp.where(jnp.abs(u_cyclo) <= 10, u_cyclo, jnp.nan)
    v_cyclo = jnp.where(jnp.abs(v_cyclo) <= 10, v_cyclo, jnp.nan)
    u_geos = jnp.where(jnp.abs(u_geos) <= 10, u_geos, jnp.nan)
    v_geos = jnp.where(jnp.abs(v_geos) <= 10, v_geos, jnp.nan)
    # to cpu
    uv_fields = {
        "Cyclogeostrophy": (np.array(u_cyclo, dtype=np.float32), np.array(v_cyclo, dtype=np.float32)),
        "Geostrophy": (np.array(u_geos, dtype=np.float32), np.array(v_geos, dtype=np.float32))
    }

    return uv_fields, lat_u, lon_u, lat_v, lon_v, mask


def _kinematics_to_dataset(
    kinematics_vars: dict,
    time: Float[Array, "time"],
    lat_t: Float[Array, "lat lon"],
    lon_t: Float[Array, "lat lon"]
) -> xr.Dataset:
    uv_coords = {
        "time": (["time"], time),
        "latitude": (["latitude"], np.unique(lat_t).astype(np.float32)),
        "longitude": (["longitude"], np.unique(lon_t).astype(np.float32))
    }
    data_vars = {}
    for key, (field, attrs) in kinematics_vars.items():
        data_vars[f"{key}"] = (["time", "latitude", "longitude"], field, attrs)

    return xr.Dataset(data_vars=data_vars, coords=uv_coords)


def _loss_to_dataset(kinematics_ds: xr.Dataset, loss_vars: dict) -> xr.Dataset:
    data_vars = {}
    for key, (field, attrs) in loss_vars.items():
        data_vars[f"{key}"] = (["time", "latitude", "longitude"], field, attrs)

    return kinematics_ds.assign(**data_vars)


def _save_all_times_dataset(
    ds: xr.Dataset, 
    experiment_data: ExperimentData, 
    ds_name: str,
    append: bool = False
) -> None:
    all_times_path = os.path.join(
        experiment_data.experiment_path, experiment_data.results_dir, f"all_times_{ds_name}.zarr"
    )
    all_times_ds = ds.copy(deep=False)
    all_times_var_names = {
        var_name: var_name.replace("\\langle ", "").replace(" \\rangle", "")  # not averaged
        for var_name in all_times_ds.data_vars
    }
    all_times_ds.rename_vars(all_times_var_names)
    if experiment_data.filesystem.exists(all_times_path):
        if append:
            all_times_ds.to_zarr(experiment_data.filesystem.get_path(all_times_path), append_dim="time")
        else:
            all_times_ds_ = xr.open_zarr(experiment_data.filesystem.get_path(all_times_path))
            for var in all_times_ds_.data_vars:
                all_times_ds_[var] += all_times_ds[var]
            all_times_ds_.to_zarr(experiment_data.filesystem.get_path(all_times_path), mode="r+")
    else:
        all_times_ds.to_zarr(experiment_data.filesystem.get_path(all_times_path))
    del all_times_ds


def process_batch(
    idx0: int,
    idx1: int,
    ssh_ds: xr.Dataset,
    cyclogeostrophy_fun: Callable,
    lat_t: Float[Array, "lat lon"],
    lon_t: Float[Array, "lat lon"],
    experiment_data: ExperimentData,
    experiment_config: str
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset, np.ndarray]:
    LOGGER.info("2.i.1. Estimating SSC - mini-batch")
    adt_t = ssh_ds.adt.isel(time=slice(idx0, idx1)).values
    uv_fields, lat_u, lon_u, lat_v, lon_v, mask = _estimate_ssc(cyclogeostrophy_fun, adt_t, lat_t, lon_t)
    sla_t = ssh_ds.sla.isel(time=slice(idx0, idx1)).values
    uva_fields, _, _, _, _, _ = _estimate_ssc(cyclogeostrophy_fun, sla_t, lat_t, lon_t)
    del adt_t, sla_t

    LOGGER.info("2.i.2. Computing loss value and grad - mini-batch")
    dx_u, dy_u = compute_spatial_step(lat_u, lon_u)
    dx_v, dy_v = compute_spatial_step(lat_v, lon_v)
    coriolis_factor_u = compute_coriolis_factor(lat_u)
    coriolis_factor_v = compute_coriolis_factor(lat_v)
    loss_vars = compute_loss_value_and_grad(
        uv_fields, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v, mask
    )
    del dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v, uv_fields, uva_fields

    LOGGER.info("2.i.3. Storing in a dataset - mini-batch")
    time = ssh_ds.time[idx0:idx1].values
    kinematics_ds = _kinematics_to_dataset({}, time, lat_t, lon_t)
    kinematics_ds = _loss_to_dataset(kinematics_ds, loss_vars)
    kinematics_ds = kinematics_ds.where(~mask)
    kinematics_ds.attrs["experiment_config"] = experiment_config

    LOGGER.info("2.i.4. Saving datasets - mini-batch")
    _save_all_times_dataset(kinematics_ds, experiment_data, "kinematics", append=True)
    del kinematics_ds, mask
