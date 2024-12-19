import logging
from collections.abc import Callable
import os
from typing import Dict, Tuple

from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.conf import HydraConf, JobConf
from hydra.experimental.callback import Callback
from hydra_zen import store, zen, make_custom_builds_fn

from experiment.evaluation._process import estimate_and_evaluate
from experiment.evaluation.visualization import plot_fields
from experiment.io import experiment_store, ssh_store
from experiment.io.experiment import ExperimentData
from experiment.io.ssh import SSHData
from experiment.logger._logger import LOGGER
from experiment.methods import cyclogeostrophy_conf
from experiment.preproc import (
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
    ]
)
def assess_cyclogeostrophy_impact(
    experiment_data: ExperimentData,  # where and how experiment description and outputs are stored (i.e. local or s3)
    ssh_data: SSHData,  # input SSH data
    ssh_rename: Dict[str, str] | None = None,  # dictionary mapping SSH dataset original to new variables names
    ssh_preproc: Callable = ssh_lon_to_180_180_preproc_conf,  # preprocessing applied to the drifters
    spatial_extent: Tuple[float, float, float, float] | None = None,  # spatial domain bounding box ([lon0, lon1, lat0, lat1])
    temporal_extent: Tuple[str, str] | None = None,  # temporal domain window
    cyclogeostrophy_fun: Callable = cyclogeostrophy_conf,  # callable for the cyclogeostrophy, with parameters set
    memory_per_device: int = 35,  # available VRAM or RAM (in Gb)
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
        ssh_rename (dict, optional): Dictionary mapping SSH dataset original to new variables names.
            Defaults to None, which means no variables will be renamed.
        ssh_preproc (Callable, optional): Function(s) applied to preprocess the SSH data.
            Defaults to ssh_lon_to_180_180_preproc_conf, which means longitudes are converted from [0, 360] to
            [-180, 180].
        spatial_extent (list[float, float, float, float], optional): Spatial domain bounding box as
            [lon0, lon1, lat0, lat1].
            Defaults to None, which means no spatial restriction.
        temporal_extent (list[str, str], optional): Temporal domain window as a pair of date strings [start, end].
            Defaults to None, which means no temporal restriction.
        cyclogeostrophy_fun (Callable, optional): Function for cyclogeostrophy computation with predefined parameters.
            Defaults to `cyclogeostrophy_conf`, which implies the use of the default `jaxparrow.cyclogeostrophy`
            parameters.
        memory_per_device (int, optional): Available VRAM or RAM per device in gigabytes (GB).
            Defaults to 35 GB.
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
    LOGGER.info("1.1. Loading SSH")
    ssh_data.open_dataset(ssh_rename)
    LOGGER.info("1.2. Preprocessing SSH")
    ssh_data.apply_preproc(ssh_preproc)
    LOGGER.info("1.3. Applying SSH extent")
    LOGGER.debug(
        f"before applying extent: "
        f"SSH domain dimensions: {ssh_data.dataset.dims}"
    )
    ssh_data.apply_extent(spatial_extent, temporal_extent)
    LOGGER.debug(
        f"after applying extent: "
        f"SSH domain dimensions: {ssh_data.dataset.dims}"
    )

    LOGGER.info("2. Estimating and evaluating SSC methods in mini-batch. This may take a while...")
    estimate_and_evaluate(
        experiment_data, experiment_config,
        ssh_data,
        cyclogeostrophy_fun,
        memory_per_device
    )


if __name__ == "__main__":
    builds = make_custom_builds_fn(populate_full_signature=True)
    store(HydraConf(job=JobConf(chdir=False), callbacks={"env_var": builds(EnvVar)}))

    store.add_to_hydra_store()

    zen(assess_cyclogeostrophy_impact).hydra_main(
        config_name="cyclogeostrophy_impact_experiment",
        version_base="1.1",
        config_path=".",
    )
