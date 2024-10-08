import os
from typing import Union

import cachier
import copernicusmarine as cm
import s3fs
import xarray as xr

from ..logger._logger import LOGGER
from ._filesystem import S3FileSystem


class Resource:
    path: Union[str, s3fs.S3Map]

    def open(self) -> xr.Dataset:
        if ".zarr" in self.path:
            ds = xr.open_zarr(self.path)
        elif "*.nc" in self.path:
            ds = xr.open_mfdataset(self.path)
        else:
            ds = xr.open_dataset(self.path)

        return ds


class LocalResource(Resource):
    def __init__(self, path: str):
        self.path = path


class S3Resource(Resource):
    def __init__(self, path: str, s3_fs: S3FileSystem):
        self.path = s3fs.S3Map(path, s3_fs._fs)


class URLResource(Resource):
    def __init__(self, url: str):
        self.path = url


class CopernicusResource:
    def __init__(
            self,
            cms_dataset_id: str,
            cms_username_env_var: str = "COPERNICUSMARINE_SERVICE_USERNAME",
            cms_password_env_var: str = "COPERNICUSMARINE_SERVICE_PASSWORD",
            disable_caching: bool = True
    ):
        if disable_caching:
            cachier.disable_caching()

        cm.login(os.environ[cms_username_env_var], os.environ[cms_password_env_var], overwrite_configuration_file=True)
        self.dataset_id = cms_dataset_id

    def open(self) -> xr.Dataset:
        dataset_options = {
            "dataset_id": self.dataset_id,
        }

        ds = None
        while ds is None:
            try:  # fetching the catalog might fail
                ds = cm.open_dataset(**dataset_options)
            except Exception as e:
                LOGGER.warning(f"catch exception {e}")

        return ds
