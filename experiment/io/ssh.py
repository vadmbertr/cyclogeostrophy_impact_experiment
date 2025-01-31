from collections.abc import Callable
from typing import Dict, Tuple

import numpy as np

from ._resource import CopernicusResource, Resource


class SSHData:
    def __init__(self, resource: CopernicusResource | Resource):
        self.resource = resource
        self.dataset = None

    def open_dataset(self, rename: Dict[str, str] = None):
        ds = self.resource.open()

        if rename is not None:
            ds = ds.rename(rename)

        self.dataset = ds

    def apply_preproc(self, preproc_fn: Callable = None):
        ds = self.dataset

        if preproc_fn is not None:
            ds = preproc_fn(ds)

        self.dataset = ds[["adt", "sla"]]

    def apply_extent(
        self, 
        spatial_extent: Tuple[float, float, float, float] = None, 
        temporal_extent: Tuple[str, str] = None
    ):
        ds = self.dataset

        if spatial_extent is not None:
            ds = ds.sel(
                longitude=slice(spatial_extent[0], spatial_extent[1]),
                latitude=slice(spatial_extent[2], spatial_extent[3])
            )

        if temporal_extent is not None:
            ds = ds.sel(
                time=slice(np.datetime64(temporal_extent[0]), np.datetime64(temporal_extent[1]))
            )

        self.dataset = ds
