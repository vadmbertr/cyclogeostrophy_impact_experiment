import os

from cartopy import crs as ccrs
import cmocean.cm as cmo
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..io.experiment import ExperimentData
from ..logger._logger import LOGGER


AMP_CMAP = cmo.amp  # noqa
BALANCE_R_CMAP = cmo.balance_r  # noqa
CURL_CMAP = cmo.curl  # noqa
DELTA_CMAP = cmo.delta  # noqa
DENSE_CMAP = cmo.dense  # noqa
HALINE_CMAP = cmo.haline  # noqa
MATTER_CMAP = cmo.matter  # noqa
SPEED_CMAP = cmo.speed  # noqa


def _plot(
        ssc_fields_ds: xr.Dataset, data_var: str,
        cmap: mpl.colors.LinearSegmentedColormap, vmax: float, vmin: float = None, cmap_centered: bool = False
) -> plt.Figure:
    if vmax is None:
        vmax = _get_max(ssc_fields_ds, [data_var], apply_abs=cmap_centered)
    if cmap_centered:
        vmin = -vmax
    elif vmin is None:
        vmin = 0

    field = ssc_fields_ds[data_var]

    fig, ax = plt.subplots(figsize=(13, 4), subplot_kw={"projection": ccrs.PlateCarree()})

    ax.set_title(field.attrs.get("method", ""))
    im = ax.pcolormesh(ssc_fields_ds.longitude, ssc_fields_ds.latitude, field,
                       cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(),
                       linewidth=0, rasterized=True)
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])  # noqa
    clb = fig.colorbar(im, cax=cax, label=field.attrs.get("what", data_var))
    clb.ax.set_title(field.attrs.get("units", ""))
    ax.coastlines()

    return fig


def _save_fig(
        ssc_fields_ds: xr.Dataset, data_var: str,
        experiment_data: ExperimentData, plot_path: str,
        cmap: mpl.colors.LinearSegmentedColormap, vmax: float = None, vmin: float = None, cmap_centered: bool = False
):
    try:
        fig = _plot(ssc_fields_ds, data_var, cmap, vmax, vmin, cmap_centered)
        filepath = os.path.join(plot_path, f"{data_var}.pdf")
        try:
            f = experiment_data.filesystem.open(filepath, "wb")
            fig.savefig(f, format="pdf")
            f.close()
        except Exception as e:
            LOGGER.warning(e)
            fig.savefig(filepath, format="pdf")
        plt.close()
    except Exception as e:
        LOGGER.warning(e)


def _get_max(ssc_fields_ds: xr.Dataset, data_vars: [], apply_abs: bool = False) -> float:
    try:
        if apply_abs:
            ds = abs(ssc_fields_ds[data_vars])
        else:
            ds = ssc_fields_ds[data_vars]
        vmax = float(ds.max().to_dataarray().max())
    except Exception as e:
        LOGGER.warning(e)
        vmax = None
    return vmax


def _get_quantile(ssc_fields_ds: xr.Dataset, data_vars: [], quantile: float, apply_abs: bool = False) -> float:
    try:
        if apply_abs:
            ds = abs(ssc_fields_ds[data_vars])
        else:
            ds = ssc_fields_ds[data_vars]
        data = ds.to_array().values.ravel()
        vmax = np.quantile(data[np.isfinite(data)], quantile)
    except Exception as e:
        LOGGER.warning(e)
        vmax = None
    return vmax


def _get_vars(ssc_fields_ds: xr.Dataset, contains: str, excludes: str = None) -> []:
    data_vars = ssc_fields_ds.data_vars
    return [
        data_var for data_var in data_vars
        if (contains in data_var) and ((excludes is None) or (excludes not in data_var))
    ]


def _plot_kinematics(
    ssc_fields_ds: xr.Dataset,
    experiment_data: ExperimentData,
    plot_path: str,
    quantile: float = None
):
    u_vars = _get_vars(ssc_fields_ds, contains="u_", excludes="_diff_")
    v_vars = _get_vars(ssc_fields_ds, contains="v_", excludes="_diff_")
    magn_vars = _get_vars(ssc_fields_ds, contains="magn_", excludes="_diff_")
    nrv_vars = _get_vars(ssc_fields_ds, contains="nrv_", excludes="_diff_")
    eke_vars = _get_vars(ssc_fields_ds, contains="eke_", excludes="_diff_")

    if quantile is None:
        vmax_fn = _get_max
    else:
        vmax_fn = lambda f, v, apply_abs=False: _get_quantile(f, v, quantile, apply_abs)

    u_max = vmax_fn(ssc_fields_ds, u_vars)
    v_max = vmax_fn(ssc_fields_ds, v_vars)
    magn_max = vmax_fn(ssc_fields_ds, magn_vars)
    nrv_max = vmax_fn(ssc_fields_ds, nrv_vars, apply_abs=True)
    eke_max = vmax_fn(ssc_fields_ds, eke_vars)

    for data_var in u_vars:
        _save_fig(ssc_fields_ds, data_var, experiment_data, plot_path, DELTA_CMAP, vmax=u_max, cmap_centered=True)
    for data_var in v_vars:
        _save_fig(ssc_fields_ds, data_var, experiment_data, plot_path, DELTA_CMAP, vmax=v_max, cmap_centered=True)
    for data_var in magn_vars:
        _save_fig(ssc_fields_ds, data_var, experiment_data, plot_path, SPEED_CMAP, vmax=magn_max)
    for data_var in nrv_vars:
        _save_fig(ssc_fields_ds, data_var, experiment_data, plot_path, CURL_CMAP, vmax=nrv_max, cmap_centered=True)
    for data_var in eke_vars:
        _save_fig(ssc_fields_ds, data_var, experiment_data, plot_path, MATTER_CMAP, vmax=eke_max)


def _plot_errors(ssc_fields_ds: xr.Dataset, experiment_data: ExperimentData, plot_path: str, quantile: float = None):
    err_u_vars = _get_vars(ssc_fields_ds, contains="err_u_", excludes="_diff_")
    err_v_vars = _get_vars(ssc_fields_ds, contains="err_v_", excludes="_diff_")
    err_vars = _get_vars(ssc_fields_ds, contains="err_", excludes="_diff_")

    if quantile is None:
        vmax_fn = _get_max
    else:
        vmax_fn = lambda f, v=False: _get_quantile(f, v, quantile)

    err_u_max = vmax_fn(ssc_fields_ds, err_vars)
    err_v_max = vmax_fn(ssc_fields_ds, err_vars)
    err_max = vmax_fn(ssc_fields_ds, err_vars)

    for data_var in err_u_vars:
        _save_fig(ssc_fields_ds, data_var, experiment_data, plot_path, AMP_CMAP, vmax=err_u_max)
    for data_var in err_v_vars:
        _save_fig(ssc_fields_ds, data_var, experiment_data, plot_path, AMP_CMAP, vmax=err_v_max)
    for data_var in err_vars:
        _save_fig(ssc_fields_ds, data_var, experiment_data, plot_path, AMP_CMAP, vmax=err_max)


def _plot_obs_density(ssc_fields_ds: xr.Dataset, experiment_data: ExperimentData, plot_path: str):
    obs_density_vars = _get_vars(ssc_fields_ds, contains="obs_density")

    obs_density_max = _get_max(ssc_fields_ds, obs_density_vars)

    for data_var in obs_density_vars:
        _save_fig(ssc_fields_ds, data_var, experiment_data, plot_path, DENSE_CMAP, vmax=obs_density_max)


def _plot_differences(
    errors_ds: xr.Dataset,
    kinematics_ds: xr.Dataset,
    experiment_data: ExperimentData,
    plot_path: str,
    quantile: float = None
):
    def do_plot(contains: str, vmax: float = None):
        if "_rel_" in contains:
            excludes = None
        else:
            excludes = "_rel_"
        u_vars = _get_vars(kinematics_ds, contains=f"u{contains}", excludes=excludes)
        v_vars = _get_vars(kinematics_ds, contains=f"v{contains}", excludes=excludes)
        magn_vars = _get_vars(kinematics_ds, contains=f"magn{contains}", excludes=excludes)
        nrv_vars = _get_vars(kinematics_ds, contains=f"nrv{contains}", excludes=excludes)
        eke_vars = _get_vars(kinematics_ds, contains=f"eke{contains}", excludes=excludes)
        err_u_vars = _get_vars(errors_ds, contains=f"err_u{contains}", excludes=excludes)
        err_v_vars = _get_vars(errors_ds, contains=f"err_v{contains}", excludes=excludes)
        err_vars = _get_vars(errors_ds, contains=f"err{contains}", excludes=excludes)

        for data_var in u_vars + v_vars + magn_vars + nrv_vars + eke_vars:
            if quantile is not None:
                vmax = _get_quantile(kinematics_ds, [data_var], quantile, apply_abs=True)
            _save_fig(
                kinematics_ds, data_var, experiment_data, plot_path, BALANCE_R_CMAP, vmax=vmax, cmap_centered=True
            )

        for data_var in err_u_vars + err_v_vars + err_vars:
            if quantile is not None:
                vmax = _get_quantile(errors_ds, [data_var], quantile, apply_abs=True)
            _save_fig(
                errors_ds, data_var, experiment_data, plot_path, BALANCE_R_CMAP, vmax=vmax, cmap_centered=True
            )

    do_plot(contains="_diff_")
    do_plot(contains="_diff_rel_")


def _plot_ssh(ssh_fields_ds: xr.Dataset, experiment_data: ExperimentData, plot_path: str):
    data_var = "adt"

    vmax = np.nanmax(ssh_fields_ds[data_var])
    vmin = np.nanmin(ssh_fields_ds[data_var])

    _save_fig(ssh_fields_ds, data_var, experiment_data, plot_path, HALINE_CMAP, vmax=vmax, vmin=vmin)


def plot_fields(errors_ds: xr.Dataset, kinematics_ds: xr.Dataset, experiment_data: ExperimentData):
    plot_path = os.path.join(experiment_data.experiment_path, experiment_data.results_dir, "plots")
    experiment_data.filesystem.makedirs(plot_path)

    _plot_kinematics(kinematics_ds, experiment_data, plot_path, quantile=.999)  # noqa
    _plot_errors(errors_ds, experiment_data, plot_path, quantile=.999)  # noqa
    _plot_obs_density(errors_ds, experiment_data, plot_path)  # noqa
    _plot_differences(errors_ds, kinematics_ds, experiment_data, plot_path, quantile=.999)  # noqa
    _plot_ssh(kinematics_ds, experiment_data, plot_path)  # noqa
