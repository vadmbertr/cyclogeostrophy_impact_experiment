import os
from typing import List

from cartopy import crs as ccrs
import cmocean.cm as cmo
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..io.experiment import ExperimentData
from ..logger._logger import LOGGER


AMP_CMAP = cmo.amp
BALANCE_R_CMAP = cmo.balance_r
CURL_CMAP = cmo.curl
DELTA_CMAP = cmo.delta
DENSE_CMAP = cmo.dense
HALINE_CMAP = cmo.haline
MATTER_CMAP = cmo.matter
SPEED_CMAP = cmo.speed


def _plot(
    da: xr.DataArray, data_var: str,
    cmap: mpl.colors.LinearSegmentedColormap, vmax: float, vmin: float = None, cmap_centered: bool = False,
    time: str = None
) -> plt.Figure:
    if vmax is None:
        vmax = _get_max(da, apply_abs=cmap_centered)
        clb_extend = "neither"
    else:
        clb_extend = "both"
    if cmap_centered:
        vmin = -vmax
    elif vmin is None:
        vmin = 0
        if clb_extend == "both":
            clb_extend = "max"

    fig, ax = plt.subplots(figsize=(13, 4), subplot_kw={"projection": ccrs.PlateCarree()})

    if time is not None:
        fig.suptitle(time)

    ax.set_title(da.attrs.get("method", ""))
    im = ax.pcolormesh(
        da.longitude, da.latitude, da,
        cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(),
        linewidth=0, rasterized=True
    )
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    clb = fig.colorbar(im, cax=cax, label=da.attrs.get("what", data_var), extend=clb_extend)
    clb.ax.set_title(da.attrs.get("units", ""))
    ax.coastlines()

    return fig


def _save_fig(
    da: xr.DataArray, data_var: str,
    experiment_data: ExperimentData, plot_path: str,
    cmap: mpl.colors.LinearSegmentedColormap, vmax: float = None, vmin: float = None, cmap_centered: bool = False,
    all_times: bool = False
):
    def save_fig(_da: xr.DataArray, _filename: str, _time: str = None):
        try:
            fig = _plot(_da, data_var, cmap, vmax, vmin, cmap_centered, _time)
            filepath = os.path.join(plot_path, _filename)
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

    if all_times:
        for time in da.time:
            time_str = str(time.values)[:-10]
            save_fig(da.sel(time=time), f"{data_var}_{time_str}.pdf", time_str)
    else:
        save_fig(da, f"{data_var}.pdf")


def _get_max(da: xr.DataArray, apply_abs: bool = False) -> float:
    try:
        if apply_abs:
            da = abs(da)
        vmax = float(da.max())
    except Exception as e:
        LOGGER.warning(e)
        vmax = None
    return vmax


def _get_quantile(ds: xr.Dataset, data_vars: List[str], quantile: float, apply_abs: bool = False) -> float:
    try:
        if apply_abs:
            ds = abs(ds[data_vars])
        else:
            ds = ds[data_vars]
        data = ds.to_array().values.ravel()
        vmax = np.quantile(data[np.isfinite(data)], quantile)
    except Exception as e:
        LOGGER.warning(e)
        vmax = None
    return vmax


def _get_vars(ds: xr.Dataset, contains: str, excludes: str = None) -> List[str]:
    data_vars = ds.data_vars
    return [
        data_var for data_var in data_vars
        if (contains in data_var) and ((excludes is None) or (excludes not in data_var))
    ]


def _plot_kinematics(
    kinematics_ds: xr.Dataset,
    experiment_data: ExperimentData,
    plot_path: str,
    quantile: float = None, 
    all_times: bool = False
):
    u_vars = _get_vars(kinematics_ds, contains="u_", excludes="_diff_")
    v_vars = _get_vars(kinematics_ds, contains="v_", excludes="_diff_")
    magn_vars = _get_vars(kinematics_ds, contains="magn_", excludes="_diff_")
    nrv_vars = _get_vars(kinematics_ds, contains="nrv_", excludes="_diff_")
    eke_vars = _get_vars(kinematics_ds, contains="eke_", excludes="_diff_")

    v_vars = list(set(v_vars) - set(nrv_vars))  # exclude "nrv_*" from v_vars

    if quantile is None:
        vmax_fn = lambda *_: None
    else:
        vmax_fn = lambda f, v, apply_abs=False: _get_quantile(f, v, quantile, apply_abs)

    u_max = vmax_fn(kinematics_ds, u_vars, apply_abs=True)
    v_max = vmax_fn(kinematics_ds, v_vars, apply_abs=True)
    magn_max = vmax_fn(kinematics_ds, magn_vars)
    nrv_max = vmax_fn(kinematics_ds, nrv_vars, apply_abs=True)
    eke_max = vmax_fn(kinematics_ds, eke_vars)

    for data_var in u_vars:
        _save_fig(
            kinematics_ds[data_var], data_var, 
            experiment_data, plot_path, 
            DELTA_CMAP, vmax=u_max, cmap_centered=True, 
            all_times=all_times
        )
    for data_var in v_vars:
        _save_fig(
            kinematics_ds[data_var], data_var, 
            experiment_data, plot_path, 
            DELTA_CMAP, vmax=v_max, cmap_centered=True, 
            all_times=all_times
        )
    for data_var in magn_vars:
        _save_fig(
            kinematics_ds[data_var], data_var, 
            experiment_data, plot_path, 
            SPEED_CMAP, vmax=magn_max, 
            all_times=all_times
        )
    for data_var in nrv_vars:
        _save_fig(
            kinematics_ds[data_var], data_var, 
            experiment_data, plot_path, 
            CURL_CMAP, vmax=nrv_max, cmap_centered=True, 
            all_times=all_times
        )
    for data_var in eke_vars:
        _save_fig(
            kinematics_ds[data_var], data_var, 
            experiment_data, plot_path, 
            MATTER_CMAP, vmax=eke_max, 
            all_times=all_times
        )


def _plot_errors(errors_ds: xr.Dataset, experiment_data: ExperimentData, plot_path: str, quantile: float = None):
    err_u_vars = _get_vars(errors_ds, contains="err_u_", excludes="_diff_")
    err_v_vars = _get_vars(errors_ds, contains="err_v_", excludes="_diff_")
    err_vars = _get_vars(errors_ds, contains="err_", excludes="_diff_")

    if quantile is None:
        vmax_fn = lambda *args: None
    else:
        vmax_fn = lambda f, v=False: _get_quantile(f, v, quantile)

    err_u_max = vmax_fn(errors_ds, err_vars)
    err_v_max = vmax_fn(errors_ds, err_vars)
    err_max = vmax_fn(errors_ds, err_vars)

    for data_var in err_u_vars:
        _save_fig(errors_ds[data_var], data_var, experiment_data, plot_path, AMP_CMAP, vmax=err_u_max)
    for data_var in err_v_vars:
        _save_fig(errors_ds[data_var], data_var, experiment_data, plot_path, AMP_CMAP, vmax=err_v_max)
    for data_var in err_vars:
        _save_fig(errors_ds[data_var], data_var, experiment_data, plot_path, AMP_CMAP, vmax=err_max)


def _plot_obs_density(errors_ds: xr.Dataset, experiment_data: ExperimentData, plot_path: str):
    obs_density_vars = _get_vars(errors_ds, contains="obs_density")

    for data_var in obs_density_vars:
        _save_fig(errors_ds[data_var], data_var, experiment_data, plot_path, DENSE_CMAP)


def _plot_differences(
    errors_ds: xr.Dataset | None,
    kinematics_ds: xr.Dataset,
    experiment_data: ExperimentData,
    plot_path: str,
    quantile: float = None, 
    all_times: bool = False
):
    def do_plot(contains: str, vmax: float = None):
        if "_rel_" in contains:
            excludes = None
        else:
            excludes = "_rel_"

        if errors_ds is not None:
            err_u_vars = _get_vars(errors_ds, contains=f"err_u{contains}", excludes=excludes)
            err_v_vars = _get_vars(errors_ds, contains=f"err_v{contains}", excludes=excludes)
            err_vars = _get_vars(errors_ds, contains=f"err{contains}", excludes=excludes)

            for data_var in err_u_vars + err_v_vars + err_vars:
                if quantile is not None:
                    vmax = _get_quantile(errors_ds, [data_var], quantile, apply_abs=True)
                _save_fig(
                    errors_ds[data_var], data_var, 
                    experiment_data, plot_path, 
                    BALANCE_R_CMAP, vmax=vmax, cmap_centered=True
                )
        
        u_vars = _get_vars(kinematics_ds, contains=f"u{contains}", excludes=excludes)
        v_vars = _get_vars(kinematics_ds, contains=f"v{contains}", excludes=excludes)
        magn_vars = _get_vars(kinematics_ds, contains=f"magn{contains}", excludes=excludes)
        nrv_vars = _get_vars(kinematics_ds, contains=f"nrv{contains}", excludes=excludes)
        eke_vars = _get_vars(kinematics_ds, contains=f"eke{contains}", excludes=excludes)

        v_vars = list(set(v_vars) - set(nrv_vars))  # exclude "nrv_*" from v_vars

        for data_var in u_vars + v_vars + magn_vars + nrv_vars + eke_vars:
            if quantile is not None:
                vmax = _get_quantile(kinematics_ds, [data_var], quantile, apply_abs=True)
            _save_fig(
                kinematics_ds[data_var], data_var, 
                experiment_data, plot_path, 
                BALANCE_R_CMAP, vmax=vmax, cmap_centered=True, 
                all_times=all_times
            )

    do_plot(contains="_diff_")
    do_plot(contains="_diff_rel_")


def _plot_ssh(
    kinematics_ds: xr.Dataset, 
    experiment_data: ExperimentData, 
    plot_path: str, 
    quantile: float = None,
    all_times: bool = False
):
    data_var = "adt"

    if quantile is not None:
        vmax = _get_quantile(kinematics_ds, [data_var], quantile)
        vmin = _get_quantile(kinematics_ds, [data_var], 1 - quantile)
    else:
        vmax = None
        vmin = None

    _save_fig(
        kinematics_ds[data_var], data_var, 
        experiment_data, plot_path, 
        HALINE_CMAP, vmax=vmax, vmin=vmin, 
        all_times=all_times
    )


def plot_fields(
    errors_ds: xr.Dataset | None, 
    kinematics_ds: xr.Dataset, 
    experiment_data: ExperimentData, 
    all_times: bool = False
):
    if all_times:
        plot_path = os.path.join(experiment_data.experiment_path, experiment_data.results_dir, "plots", "all_times")
    else:
        plot_path = os.path.join(experiment_data.experiment_path, experiment_data.results_dir, "plots", "time_averaged")
    experiment_data.filesystem.makedirs(plot_path)

    if errors_ds is not None:
        _plot_errors(errors_ds, experiment_data, plot_path, quantile=.999)
        _plot_obs_density(errors_ds, experiment_data, plot_path)
    _plot_kinematics(kinematics_ds, experiment_data, plot_path, quantile=.999, all_times=all_times)
    _plot_differences(errors_ds, kinematics_ds, experiment_data, plot_path, quantile=.999, all_times=all_times)
