from collections.abc import Callable

import clouddrift as cd
import numpy as np

from ._resource import CopernicusResource, Resource


class DrifterData:
    def __init__(self, resource: CopernicusResource | Resource):
        self.resource = resource
        self.dataset = None

    def open_dataset(
        self,
        rename: dict,
        spatial_extent: [float, float, float, float] = None,
        temporal_extent: [str, str] = None
    ):
        ds = self.resource.open()

        if rename is not None:
            ds = ds.rename(rename)

        if spatial_extent is not None:
            ds = cd.ragged.subset(
                ds,
                {
                    "lon": lambda lon: ((lon >= spatial_extent[0]) & (lon <= spatial_extent[1])),
                    "lat": lambda lat: ((lat >= spatial_extent[2]) & (lat <= spatial_extent[3])),
                },
                row_dim_name="traj"
            )

        if temporal_extent is not None:
            ds = cd.ragged.subset(
                ds,
                {
                    "time": lambda time: (
                        (time >= np.datetime64(temporal_extent[0])) &
                        (time <= np.datetime64(temporal_extent[1]))
                    )
                },
                row_dim_name="traj"
            )

        self.dataset = ds

    def apply_preproc(self, preproc_fn: Callable = None):
        ds = self.dataset

        if preproc_fn is not None:
            ds = preproc_fn(ds)

        self.dataset = ds[["rowsize", "lon", "lat", "time", "ve", "vn"]]
