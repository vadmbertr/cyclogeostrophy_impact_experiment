from hydra_zen import store, make_custom_builds_fn

from ._filesystem import LocalFileSystem, S3FileSystem
from ._resource import CopernicusResource, LocalResource, S3Resource, URLResource
from .drifter import DrifterData
from .experiment import ExperimentData
from .ssh import SSHData


__all__ = ["s3_fs_conf", "resource_store", "experiment_store", "drifter_store", "ssh_store"]

builds = make_custom_builds_fn(populate_full_signature=True)

# s3_fs_conf
local_fs_conf = builds(LocalFileSystem)
s3_fs_conf = builds(S3FileSystem)

# resource_store
copernicus_resource_conf = builds(CopernicusResource)
local_resource_conf = builds(LocalResource)
s3_resource_conf = builds(S3Resource, s3_fs=s3_fs_conf)
url_resource_conf = builds(URLResource)

resource_store = store(group="resource")
resource_store(copernicus_resource_conf, name="copernicus")
resource_store(local_resource_conf, name="local")
resource_store(s3_resource_conf, name="s3")
resource_store(url_resource_conf, name="url")

# experiment_store
local_experiment = builds(ExperimentData, filesystem=local_fs_conf)
s3_experiment = builds(ExperimentData, filesystem=s3_fs_conf)

experiment_store = store(group="experiment_data")
experiment_store(local_experiment, name="local")
experiment_store(s3_experiment, name="s3")

# drifter_store
drifter_local_conf = builds(DrifterData, resource=local_resource_conf)
drifter_s3_conf = builds(DrifterData, resource=s3_resource_conf)
drifter_url_conf = builds(DrifterData, resource=url_resource_conf)

drifter_store = store(group="drifter_data")
drifter_store(drifter_local_conf, name="local")
drifter_store(drifter_s3_conf, name="s3")
drifter_store(drifter_url_conf, name="url")

# ssh_store
ssh_copernicus_conf = builds(SSHData, resource=copernicus_resource_conf)
ssh_local_conf = builds(SSHData, resource=local_resource_conf)
ssh_s3_conf = builds(SSHData, resource=s3_resource_conf)
ssh_url_conf = builds(SSHData, resource=url_resource_conf)

ssh_store = store(group="ssh_data")
ssh_store(ssh_copernicus_conf, name="copernicus")
ssh_store(ssh_local_conf, name="local")
ssh_store(ssh_s3_conf, name="s3")
ssh_store(ssh_url_conf, name="url")
