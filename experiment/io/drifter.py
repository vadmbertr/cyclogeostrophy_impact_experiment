from collections.abc import Callable
from typing import Dict

import clouddrift as cd
import numpy as np

from ._resource import CopernicusResource, Resource
from .ssh import SSHData


class DrifterData:
    def __init__(self, resource: CopernicusResource | Resource):
        self.resource = resource
        self.dataset = None

    def open_dataset(
        self,
        rename: Dict[str, str] = None,
        ssh_data: SSHData = None,
    ):
        ds = self.resource.open()

        if rename is not None:
            ds = ds.rename(rename)

        if ssh_data is not None:
            start_time = ssh_data.dataset["time"].min().values
            end_time = ssh_data.dataset["time"].max().values
            min_lon = ssh_data.dataset["longitude"].min().values
            max_lon = ssh_data.dataset["longitude"].max().values
            min_lat = ssh_data.dataset["latitude"].min().values
            max_lat = ssh_data.dataset["latitude"].max().values

            ds = cd.ragged.subset(
                ds,
                {
                    "time": lambda time: (time >= start_time) & (time <= end_time),
                    "lon": lambda lon: (lon >= min_lon) & (lon <= max_lon),
                    "lat": lambda lat: (lat >= min_lat) & (lat <= max_lat)
                },
                row_dim_name="traj"
            )

        self.dataset = ds

    def apply_preproc(self, preproc_fn: Callable = None):
        ds = self.dataset

        if preproc_fn is not None:
            ds = preproc_fn(ds)

        self.dataset = ds[["rowsize", "lon", "lat", "time", "ve", "vn"]]
