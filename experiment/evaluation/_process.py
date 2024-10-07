from collections.abc import Callable
import math
import os

import clouddrift as cd
import jax.numpy as jnp
import numpy as np
from jaxparrow.tools.geometry import compute_coriolis_factor, compute_spatial_step
from jaxtyping import Array, Float
import xarray as xr

from ..io.drifter import DrifterData
from ..io.experiment import ExperimentData
from ..io.ssh import SSHData
from ..logger._logger import LOGGER
from .comparison import compare_methods
from .interpolation import interpolate_drifters_location
from .kinematics import compute_kinematics
from .loss import compute_loss_value_and_grad
from .metrics import compute_along_traj_metrics, compute_binned_metrics


def _estimate_batch_indices(n_time: int, n_lat: int, n_lon: int, memory_per_device: float) -> list:
    f32_size = 4  # mostly manipulate f32 arrays
    comp_mem_per_time = f32_size * n_lat * n_lon * 1e-9  # in Gb
    comp_mem_per_time *= 40  # empirical factor preventing OOM errors
    batch_size = int(memory_per_device // comp_mem_per_time)  # conservative batch size
    n_batches = math.ceil(n_time / batch_size)  # conservative number of batches
    indices = jnp.arange(1, n_batches) * batch_size
    return indices.tolist()


def _add_ssh(kinematics_ds: xr.Dataset, ssh_ds: xr.Dataset) -> xr.Dataset:
    kinematics_ds["adt"] = (
        ["latitude", "longitude"],
        ssh_ds.adt.mean(dim="time").values,
        {"what": ssh_ds.adt.attrs["long_name"], "units": "$m$"}
    )
    return kinematics_ds


def estimate_and_evaluate(
    experiment_data: ExperimentData,
    experiment_config: str,
    drifter_data: DrifterData,
    ssh_data: SSHData,
    cyclogeostrophy_fun: Callable,
    bin_size: int,
    save_all_times: bool,
    memory_per_device: int
) -> (xr.Dataset, xr.Dataset):
    ssh_ds = ssh_data.dataset

    # fix over batches
    lon_t, lat_t = jnp.meshgrid(ssh_ds.longitude.values, ssh_ds.latitude.values)

    # estimate batch indices based on domain dimensions
    n_time, n_lat, n_lon = tuple(ssh_ds.sizes.values())
    batch_indices = _estimate_batch_indices(n_time, n_lat, n_lon, memory_per_device)
    batch_indices = [0] + batch_indices + [n_time]

    # apply per batch
    errors_sum_ds, errors_count_ds, kinematics_sum_ds, kinematics_count_ds, mask = None, None, None, None, None
    for idx0, idx1 in zip(batch_indices[:-1], batch_indices[1:]):
        LOGGER.debug(f"2.i.0. mini-batch {[idx0, idx1]}")
        errors_sum_ds_batch, errors_count_ds_batch, kinematics_sum_ds_batch, kinematics_count_ds_batch, mask_batch = (
            process_batch(
                idx0, idx1,
                drifter_data.dataset, ssh_ds,
                cyclogeostrophy_fun,
                lat_t, lon_t,
                bin_size,
                save_all_times,
                experiment_data, experiment_config
            )
        )

        if errors_sum_ds_batch is not None:
            if errors_sum_ds is None:
                errors_sum_ds = errors_sum_ds_batch
            else:
                errors_sum_ds += errors_sum_ds_batch

        if errors_count_ds_batch is not None:
            if errors_count_ds is None:
                errors_count_ds = errors_count_ds_batch
            else:
                errors_count_ds += errors_count_ds_batch

        if kinematics_sum_ds is None:
            kinematics_sum_ds = kinematics_sum_ds_batch
        else:
            kinematics_sum_ds += kinematics_sum_ds_batch

        if kinematics_count_ds is None:
            kinematics_count_ds = kinematics_count_ds_batch
        else:
            kinematics_count_ds += kinematics_count_ds_batch

        if mask is None:
            mask = mask_batch
        else:
            mask |= mask_batch

    LOGGER.info("2.N.1. Normalizing batched datasets")
    with xr.set_options(keep_attrs=True):
        errors_ds = errors_sum_ds / errors_count_ds
    errors_ds["obs_density"] = errors_count_ds.to_array().mean(dim="variable")
    errors_ds["obs_density"].attrs = {"what": "Drifter observations frequency"}
    del errors_sum_ds, errors_count_ds
    with xr.set_options(keep_attrs=True):
        kinematics_ds = kinematics_sum_ds / kinematics_count_ds
    del kinematics_sum_ds, kinematics_count_ds

    LOGGER.info("2.N.2. Comparing methods")
    errors_ds, kinematics_ds = compare_methods(errors_ds, kinematics_ds)

    LOGGER.info("2.N.3. Adding mean SSH")
    kinematics_ds = _add_ssh(kinematics_ds, ssh_ds)

    LOGGER.info("2.N.4. Applying spatial mask")
    kinematics_ds = kinematics_ds.where(~mask)

    errors_ds.attrs["experiment_config"] = experiment_config
    kinematics_ds.attrs["experiment_config"] = experiment_config

    return errors_ds, kinematics_ds


def _estimate_ssc(
    cyclogeostrophy_fun: Callable,
    ssh_t: Float[Array, "time lat lon"],
    lat_t: Float[Array, "lat lon"],
    lon_t: Float[Array, "lat lon"]
) -> (
    dict,
    Float[Array, "lat lon"],
    Float[Array, "lat lon"],
    Float[Array, "lat lon"],
    Float[Array, "lat lon"],
    Float[Array, "time lat lon"]
):
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


def process_batch(
    idx0: int,
    idx1: int,
    drifter_ds: xr.Dataset,
    ssh_ds: xr.Dataset,
    cyclogeostrophy_fun: Callable,
    lat_t: Float[Array, "lat lon"],
    lon_t: Float[Array, "lat lon"],
    bin_size: int,
    save_all_times: bool,
    experiment_data: ExperimentData,
    experiment_config: str
) -> (xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset, np.ndarray):
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
    del dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v

    LOGGER.info("2.i.3. Interpolating SSC velocities to drifters positions - mini-batch")
    time = ssh_ds.time[idx0:idx1].values
    drifter_ds.time.load()
    drifter_batch = cd.ragged.subset(
        drifter_ds,
        {"time": (time[0], time[-1])},
        row_dim_name="traj"
    )

    if drifter_batch:
        drifter_batch = drifter_batch.drop_vars(["rowsize", "id"])
        drifter_batch = interpolate_drifters_location(drifter_batch, time, lat_v, lon_u, uv_fields)

        LOGGER.info("2.i.4. Evaluating SSC against drifters velocities - mini-batch")
        LOGGER.info("2.i.4.1. Computing along trajectories metrics - mini-batch")
        traj_metrics = compute_along_traj_metrics(drifter_batch, uv_fields.keys())
        del drifter_batch
        LOGGER.info("2.i.4.2. Computing binned metrics - mini-batch")
        errors_sum_ds, errors_count_ds = compute_binned_metrics(ssh_ds, traj_metrics, uv_fields.keys(), bin_size)
        del traj_metrics
    else:
        errors_sum_ds, errors_count_ds = None, None

    LOGGER.info("2.i.5. Computing additional kinematics - mini-batch")
    kinematics_vars = compute_kinematics(uv_fields, uva_fields, lat_u, lon_u, lat_v, lon_v, mask)
    del uv_fields, uva_fields

    LOGGER.info("2.i.6. Storing in a dataset - mini-batch")
    kinematics_ds = _kinematics_to_dataset(kinematics_vars, time, lat_t, lon_t)
    kinematics_ds = _loss_to_dataset(kinematics_ds, loss_vars)
    kinematics_ds = kinematics_ds.where(~mask)

    LOGGER.info("2.i.7. Comparing methods - mini-batch")
    _, kinematics_ds = compare_methods(None, kinematics_ds)  # noqa
    kinematics_ds.attrs["experiment_config"] = experiment_config

    if save_all_times:
        LOGGER.info("2.i.8. Saving dataset - mini-batch")
        kinematics_path = os.path.join(
            experiment_data.experiment_path, experiment_data.results_dir, "all_times_kinematics.zarr"
        )
        if experiment_data.filesystem.exists(kinematics_path):  # noqa
            kinematics_ds.to_zarr(experiment_data.filesystem.get_path(kinematics_path), append_dim="time")  # noqa
        else:
            kinematics_ds.to_zarr(experiment_data.filesystem.get_path(kinematics_path))  # noqa

    LOGGER.info("2.i.9. Summing along time - mini-batch")
    kinematics_sum_ds = kinematics_ds.sum(dim="time", skipna=True, keep_attrs=True)
    kinematics_count_ds = xr.apply_ufunc(np.isfinite, kinematics_ds).sum(dim="time")
    mask = np.any(mask, axis=0)
    del kinematics_ds

    return errors_sum_ds, errors_count_ds, kinematics_sum_ds, kinematics_count_ds, mask