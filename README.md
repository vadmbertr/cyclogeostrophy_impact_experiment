# Project description

The aim of this repo is to facilitate the launch of experiments assessing the impact of cyclogeostrophic corrections
applied to geostrophic Sea Surface Currents (SSC) derived from Sea Surface Height (SSH) data.

Geostrophic and cyclogeostrophic currents are computed using the package `jaxparrow` ([docs](https://jaxparrow.readthedocs.io/)).
The impact is evaluated by computing geostrophic and cyclogeostrophic differences between the:

- SSC, relative vorticity and Eddy Kinetic Energy (EKE),
- cyclogeostrophic imbalance,
- Euclidean distance between SSH derived SSC and drifters velocity.

Results are stored as datasets and dataframes.

One objective of this repo was to make the choice of the SSC inversion parameters and the input datasets
as flexible as possible, while keeping a clear track of the values used for each experiment.
For this it uses the `hydra` and `hydra-zen` libraries ([docs](https://mit-ll-responsible-ai.github.io/hydra-zen/)).

It can also be used to reproduce the results presented in our paper, see [paper-results-reproduction/README.md](paper-results-reproduction/README.md).

## Installation

Assuming `conda` is installed, running:

```shell
./depencies.sh
```

will create a `conda` environment and install the required dependencies.

## Usage

Thanks to `hydra-zen`, experiments are launched as a Python script.

Running:

```shell
python main.py --help
```

will provide `hydra-zen` formatted details about the experiment parameters:

```text
experiment_data (ExperimentData): Object representing the file system and structure where experiment descriptions and
    outputs are saved. It can be a local filesystem, but also a s3 bucket.
ssh_data (SSHData): Object representing the input SSH data. It can be only a path to the data file(s) on a local 
    filesystem, but it can also describe an URL, a s3 bucket, or a Copernicus Marine Service dataset.
drifter_data (DrifterData): Object representing the input drifter data. It can be only a path to the data file(s) on a 
    local filesystem, but it can also describe an URL, a s3 bucket, or a clouddrift dataset.
ssh_rename (dict, optional): Dictionary mapping SSH dataset original to new variables names.
    Defaults to None, which means no variables will be renamed.
drifter_rename (dict, optional): Dictionary mapping drifter dataset original to new variables names.
    Defaults to None, which means no variables will be renamed.
ssh_preproc (Callable, optional): Function(s) applied to preprocess the SSH data.
    Defaults to ssh_lon_to_180_180_preproc_conf, which means longitudes are converted from [0, 360] to
    [-180, 180].
drifter_preproc (Callable, optional): Function(s) applied to preprocess the drifter data.
    Defaults to drifter_default_preproc_conf, which means default preprocessing is applied.
spatial_extent (list[float, float, float, float], optional): Spatial domain bounding box as [lon0, lon1, lat0, lat1].
    Defaults to None, which means no spatial restriction.
temporal_extent (list[str, str], optional): Temporal domain window as a pair of date strings [start, end].
    Defaults to None, which means no temporal restriction.
cyclogeostrophy_fun (Callable, optional): Function for cyclogeostrophy computation with predefined parameters.
    Defaults to `cyclogeostrophy_conf`, which implies the use of the default `jaxparrow.cyclogeostrophy` parameters.
memory_per_device (int, optional): Available VRAM or RAM per device in gigabytes (GB).
    Defaults to 30 GB.
logger_level (int, optional): Logging output level (e.g., `logging.DEBUG`, `logging.INFO`, etc...).
    Defaults to `logging.DEBUG`.
```

See [Examples](#examples) for ready-to-use launch commands.

### Main objects types and classes

Experiments descriptions and results storage is represented by the class `experiment.io.experiement.ExperimentData`.
Its constructor takes as arguments:

- a `FileSystem` object,
- a `string` indicating the path on this filesystem to the root directory of the experiment outputs.

How to access input datasets is described by the class `experiment.io._resource.Resource` and its subclasses.
The subclasses allow specifying where the resource is accessed: on a local filesystem (`LocalResource`),
on the "cloud" (`URLResource`), on a s3 bucket (`S3Resource`), or through a specific provider
(Copernicus Marine Service: `CopernicusResource` or CloudDrift: `CloudDriftResource`).
`LocalResource` and `URLResource` take as argument a single `string` representing the `path` (or the URL)
 to the dataset(s).
`S3Resource` also takes a `path` as argument, plus a `S3FileSystem` object allowing to access the s3 bucket.
`CopernicusResource` does not have a `path` argument, but several `string` describing the targeted dataset
and how to connect to the Copernicus Marine Service datastore.
See details in [Pointing to input or output data](#pointing-to-input-or-output-data).

Finally, a filesystem is represented by the `experiment.io._filesystem.FileSystem` class and subclasses:
`LocalFileSystem` for local filesystem, and `S3FileSystem` for s3 buckets.
Local filesystems objects constructor do not have any argument, while the `S3FileSystem` constructor takes several
arguments to authenticate and connect to the bucket (see details in [Accessing s3 buckets](#accessing-s3-buckets)).

### Examples

Beside creating yaml configuration files for the experiments, `hydra-zen` allows to specify custom values for the
objects passed as arguments to the script with the syntax `object_group=entry_name`,
but also to their parameters with `object_group.arg=value` (not limited to one tree level).

#### Pointing to input or output data

##### Local or URL data

For example, using the `local` configuration (which is the default) for `experiment_data`, `ssh_data` and `drifter_data`
can be done with:

```shell
python main.py experiment_data=local ssh_data=local drifter_data=local
```

And prescribing the local `path` of the `ssh_data` or `drifter_data` `resource` is achieved with:

```shell
python main.py experiment_data=local ssh_data=local drifter_data=local 
  ssh_data.resource.path='fullpath_to_ssh_file.zarr' drifter_data.resource.path='fullpath_to_drifter_file.zarr'
```

It is also possible to access data from URLs as if the resource was stored locally:

```shell
python main.py experiment_data=local ssh_data=local drifter_data=local 
  ssh_data.resource.path='https://url_to_ssh_file.zarr' drifter_data.resource.path='fullpath_to_drifter_file.zarr'
```

##### s3 buckets

s3 buckets can be read and write by using the `s3` entry of the `experiment_data`, `ssh_data` and `drifter_data` groups:

```shell
python main.py experiment_data=s3 ssh_data=s3 drifter_data=s3
```

Public buckets can be accessed by setting the `anon` argument to the `S3FileSystem` constructor to `True` using:

```shell
python main.py ssh_data=s3 drifter_data=s3 ssh_data.resource.s3_fs.anon=True drifter_data.resource.s3_fs.anon=True
```

For private buckets (`anon=False`, which is the default) the application assumes that the credentials used are stored
in environment variables.
Again, the key of those environment variables can be changed using for example:

```shell
python main.py experiment_data=s3 experiment_data.filesystem.anon=False 
    experiment_data.filesystem.key_env_var='<YOUR_KEY_ENV_VAR_NAME>'
```

s3 buckets have `endpoint` (`minio.lab.dive.edito.eu` for EDITO s3 for example) that should also be specified,
either from the command line:

```shell
python main.py experiment_data=s3 experiment_data.filesystem.anon=False 
    experiment_data.filesystem.endpoint='<YOUR_S3_ENDPOINT>'
```

Or from an environment variable, named `AWS_S3_ENDPOINT` by default, but that you can override:

```shell
python main.py experiment_data=s3 experiment_data.filesystem.anon=False 
    experiment_data.filesystem.endpoint_env_var='<YOUR_ENDPOINT_ENV_VAR_NAME>'
```

See the `S3FileSystem` class definition in the module `experiment.io._filesystem.py` for details
(as script parameters and directly linked to constructors arguments).

##### copernicusmarine API

`ssh_data` can be read using `copernicusmarine` API:

```shell
python main.py ssh_data=copernicus ssh_data.resource.cms_dataset_id='<THE_DATASET_ID>'
```

#### Standard Python objects

Python objects such as `list` or `dict` can be passed as:

```shell
python main.py ssh_data.resource.path='fullpath_to_ssh_file.zarr' drifter_data.resource.path='fullpath_to_drifter_file.zarr'
  spatial_extent='[-60, -50, 25, 35]' temporal_extent='["2023-08-01", "2023-08-02"]'
```

### Input datasets requirements

The SSH and drifters data are both expected to be `xarray` ([docs](https://docs.xarray.dev/)) readable file(s)
such as `zarr` or `NetCDF` formats.
By using the wildcard `*.nc`, several `NetCDF` files can be opened at once.

SSH datasets are then processed assuming that:

- dimensions names are `time`, `latitude`, `longitude`,
- `time` is expressed in (nano)seconds since epoch,
- `latitude` and `longitude` are given in degrees,
- `longitude` range from -180 to 180,
- Absolute Dynamic Topography (ADT) and Sea Level Anomaly (SLA) are encoded in meters in the variables `adt` and `sla`.

To meet those requirements, one can use the parameter `ssh_rename` to rename the variables and dimensions, and add a
preprocessing step in the `experiment.preproc` module.
Note that the longitudes are automatically projected to [-180, 180] if provided in [0, 360] by using the transform
`(longitude + 180) % 360 - 180` which acts as the identity function if longitudes already range in [-180, 180].

Drifters data are assumed to follow `clouddrift` ([docs](https://clouddrift.org/)) ragged-array format,
meaning that they have:

- `traj` and `obs` dimensions,
- at least `lat`, `lon`, `time`, `ve` and `vn` variables,
- `time` is expressed in (nano)seconds since epoch,
- `lat` and `lon` are given in degrees,
- `lon` range from -180 to 180,
- `ve` and `vn` are in m/s,
- for the default preprocessing steps to work, the variables `location_type`, `typebuoy` and `drogue_status`
must also be present.
