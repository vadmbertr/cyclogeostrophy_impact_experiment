import logging
from collections.abc import Callable
import os
from typing import Dict, Tuple

from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.conf import HydraConf, JobConf
from hydra.experimental.callback import Callback
from hydra_zen import store, zen, make_custom_builds_fn
import xarray as xr

from experiment.evaluation._process import estimate_and_evaluate
from experiment.evaluation.visualization import plot_fields
from experiment.io import drifter_store, experiment_store, ssh_store
from experiment.io.drifter import DrifterData
from experiment.io.experiment import ExperimentData
from experiment.io.ssh import SSHData
from experiment.logger._logger import LOGGER
from experiment.methods import cyclogeostrophy_conf
from experiment.preproc import (
    drifter_preproc_store,
    drifter_default_preproc_conf,
    ssh_preproc_store,
    ssh_lon_to_180_180_preproc_conf
)


class EnvVar(Callback):
    def __init__(self, *, env_file_path: str = ".env"):
        if not os.path.exists(env_file_path):
            env_file_path = ""
        self.env_file_path = env_file_path

    def on_job_start(self, **kw):
        load_dotenv(self.env_file_path, override=True)  # load env. var. (for credentials) from file if provided


@store(
    name="cyclogeostrophy_impact_experiment",
    hydra_defaults=[
        "_self_",
        {"experiment_data": "local"},
        {"ssh_data": "local"},
        {"drifter_data": "local"},
    ]
)
def assess_cyclogeostrophy_impact(
    experiment_data: ExperimentData,  # where and how experiment description and outputs are stored (i.e. local or s3)
    ssh_data: SSHData,  # input SSH data
    drifter_data: DrifterData,  # input drifter data
    ssh_rename: Dict[str, str] | None = None,  # dictionary mapping SSH dataset original to new variables names
    drifter_rename: Dict[str, str] | None = None,  # dictionary mapping drifter dataset original to new variables names
    ssh_preproc: Callable = ssh_lon_to_180_180_preproc_conf,  # preprocessing applied to the drifters
    drifter_preproc: Callable = drifter_default_preproc_conf,  # preprocessing applied to the drifters
    spatial_extent: Tuple[float, float, float, float] | None = None,  # spatial domain bounding box ([lon0, lon1, lat0, lat1])
    temporal_extent: Tuple[str, str] | None = None,  # temporal domain window
    cyclogeostrophy_fun: Callable = cyclogeostrophy_conf,  # callable for the cyclogeostrophy, with parameters set
    bin_size: float = 1.,  # bins size for the errors computed vs. the drifters data (in °)
    do_plot: bool = True,  # whether to automatically produce plots
    do_plot_all_times: bool = False,  # whether to produce plots for each time step
    memory_per_device: int = 30,  # available VRAM or RAM (in Gb)
    logger_level: int = logging.DEBUG  # logger outputs level
):
    """
    Assess the impact of cyclogeostrophic corrections applied to geostrophic Sea Surface Currents (SSC) derived from Sea
    Surface Height (SSH) data.
    Geostrophic and cyclogeostrophic currents are computed using the package `jaxparrow`.
    The impact is evaluated by computing geostrophic and cyclogeostrophic differences between the:
        - SSC and Eddy Kinetic Energy (EKE),
        - cyclogeostrophic imbalance,
        - Euclidean distance between field SSC and drifters velocity (interpolated in the grid).
    The SSH and drifters data are expected to be `xarray` readable file(s).
    Drifters data are assumed to follow `clouddrift` ragged-array format.

    Args:
        experiment_data (ExperimentData): Object representing the file system and structure where experiment details and
            outputs are saved. It can be a local filesystem, but also a s3 bucket.
        ssh_data (SSHData): Object representing the input SSH data. It can be only a path to the data file(s) on a
            local filesystem, but it can also describe an URL, or a s3 bucket.
        drifter_data (DrifterData): Object representing the input drifter data. It can be only a path to the data
            file(s) on a local filesystem, but it can also describe an URL, or a s3 bucket.
        ssh_rename (dict, optional): Dictionary mapping SSH dataset original to new variables names.
            Defaults to None, which means no variables will be renamed.
        drifter_rename (dict, optional): Dictionary mapping drifter dataset original to new variables names.
            Defaults to None, which means no variables will be renamed.
        ssh_preproc (Callable, optional): Function(s) applied to preprocess the SSH data.
            Defaults to ssh_lon_to_180_180_preproc_conf, which means longitudes are converted from [0, 360] to
            [-180, 180].
        drifter_preproc (Callable, optional): Function(s) applied to preprocess the drifter data.
            Defaults to drifter_default_preproc_conf, which means default preprocessing is applied.
        spatial_extent (list[float, float, float, float], optional): Spatial domain bounding box as
            [lon0, lon1, lat0, lat1].
            Defaults to None, which means no spatial restriction.
        temporal_extent (list[str, str], optional): Temporal domain window as a pair of date strings [start, end].
            Defaults to None, which means no temporal restriction.
        cyclogeostrophy_fun (Callable, optional): Function for cyclogeostrophy computation with predefined parameters.
            Defaults to `cyclogeostrophy_conf`, which implies the use of the default `jaxparrow.cyclogeostrophy`
            parameters.
        bin_size (int, optional): Bin size in degrees (°) for error computations versus drifter data.
            Defaults to 1°.
        do_plot (bool, optional): Whether to automatically produce plots
            Defaults to True.
        do_plot_all_times (bool, optional): Whether to produce plots for each time step. Requires `do_plot=True`.
            Defaults to False.
        memory_per_device (int, optional): Available VRAM or RAM per device in gigabytes (GB).
            Defaults to 30 GB.
        logger_level (int, optional): Logging output level (e.g., `logging.DEBUG`, `logging.INFO`, etc...).
            Defaults to `logging.DEBUG`.

    Returns:
        None
    """

    LOGGER.setLevel(logger_level)
    # 0. init data structure
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    experiment_path = os.path.join(
        os.path.basename(os.path.dirname(hydra_output_dir)),
        os.path.basename(hydra_output_dir)
    )  # reconstruct hydra experiment path structure
    experiment_data.set_experiment_path(experiment_path)
    # 0.1. copy experiment config
    experiment_data.filesystem.copy(
        os.path.join(hydra_output_dir, ".hydra"),
        os.path.join(experiment_data.experiment_path, experiment_data.conf_dir)
    )
    # 0.2. storing experiment config
    with open(os.path.join(hydra_output_dir, ".hydra", "config.yaml"), mode="r") as f:
        experiment_config = f.read()

    LOGGER.info("1. Loading input datasets")
    LOGGER.info("1.1.1. Loading SSH")
    ssh_data.open_dataset(ssh_rename)
    LOGGER.info("1.1.2. Preprocessing SSH")
    ssh_data.apply_preproc(ssh_preproc)
    LOGGER.info("1.1.2. Applying SSH extent")
    LOGGER.debug(
        f"before applying extent: "
        f"SSH domain dimensions: {ssh_data.dataset.dims}"
    )
    ssh_data.apply_extent(spatial_extent, temporal_extent)
    LOGGER.debug(
        f"after applying extent: "
        f"SSH domain dimensions: {ssh_data.dataset.dims}"
    )
    LOGGER.info("1.2.1. Loading drifters")
    drifter_data.open_dataset(drifter_rename, ssh_data)
    LOGGER.info("1.2.2. Preprocessing drifters")
    LOGGER.debug(
        f"before preprocessing: "
        f"{int(drifter_data.dataset.traj.size)} drifters & "
        f"{int(drifter_data.dataset.obs.size)} observations"
    )
    drifter_data.apply_preproc(drifter_preproc)
    LOGGER.debug(
        f"after preprocessing: "
        f"{int(drifter_data.dataset.traj.size)} drifters & "
        f"{int(drifter_data.dataset.obs.size)} observations"
    )

    LOGGER.info("2. Estimating and evaluating SSC methods in mini-batch. This may take a while...")
    errors_ds, kinematics_ds = estimate_and_evaluate(
        experiment_data, experiment_config,
        drifter_data, ssh_data,
        cyclogeostrophy_fun,
        bin_size,
        memory_per_device
    )

    LOGGER.info("3. Saving time averaged datasets")
    errors_ds_path = os.path.join(
        experiment_data.experiment_path, experiment_data.results_dir, "time_averaged_errors.zarr"
    )
    errors_ds.to_zarr(experiment_data.filesystem.get_path(errors_ds_path))  # noqa
    kinematics_ds_path = os.path.join(
        experiment_data.experiment_path, experiment_data.results_dir, "time_averaged_kinematics.zarr"
    )
    kinematics_ds.to_zarr(experiment_data.filesystem.get_path(kinematics_ds_path))  # noqa

    if do_plot:
        LOGGER.info("4. Producing plots")
        LOGGER.info("4.1. Producing averaged time plots.")
        plot_fields(errors_ds, kinematics_ds, experiment_data)
        if do_plot_all_times:
            LOGGER.info("4.2. Producing plots for all time steps. This may take a while...")
            kinematics_ds_path = os.path.join(
                experiment_data.experiment_path, experiment_data.results_dir, "all_times_kinematics.zarr"
            )
            kinematics_ds = xr.open_zarr(experiment_data.filesystem.get_path(kinematics_ds_path))
            plot_fields(None, kinematics_ds, experiment_data, all_times=True)


if __name__ == "__main__":
    builds = make_custom_builds_fn(populate_full_signature=True)
    store(HydraConf(job=JobConf(chdir=False), callbacks={"env_var": builds(EnvVar)}))

    store.add_to_hydra_store()

    zen(assess_cyclogeostrophy_impact).hydra_main(
        config_name="cyclogeostrophy_impact_experiment",
        version_base="1.1",
        config_path=".",
    )
