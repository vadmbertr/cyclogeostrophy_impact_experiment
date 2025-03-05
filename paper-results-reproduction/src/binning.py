import numpy as np
import pandas as pd
import pyinterp
import xarray as xr


def compute_spatial_binned_errors(df: pd.DataFrame, latitude: np.ndarray, longitude: np.ndarray) -> xr.Dataset:
    binning = pyinterp.Binning2D(pyinterp.Axis(longitude + 180, is_circle=True), pyinterp.Axis(latitude))

    data_vars = {}
    for var in df.columns:
        if var in ["lat", "lon"]:
            continue
        if "_diff_" in var:
            continue

        binning.clear()
        binning.push(df["lon"] + 180, df["lat"], df[var], False)

        data_vars[f"{var}_mean"] = (
            ["latitude", "longitude"],
            binning.variable("mean").T,
            {"units": "$m/s$"}
        )
        data_vars[f"{var}_sd"] = (
            ["latitude", "longitude"],
            np.sqrt(binning.variable("variance").T),
            {"units": "$m/s$"}
        )

    data_vars["count"] = (
        ["latitude", "longitude"],
        binning.variable("count").T,
        {}
    )

    coords = {
        "latitude": (["latitude"], latitude),
        "longitude": (["longitude"], longitude)
    }
    errors_ds = xr.Dataset(data_vars=data_vars, coords=coords)

    for var in errors_ds.variables:
        if "Cyclogeostrophy" not in var:
            continue

        err_var_suffix = var.replace("_Cyclogeostrophy", "")
        err_var_geos = var.replace("_Cyclogeostrophy", "_Geostrophy")
        err_diff_var = f"{err_var_suffix}_diff_Geostrophy_Cyclogeostrophy"
        err_diff_rel_var = f"{err_var_suffix}_diff_rel_Geostrophy_Cyclogeostrophy"

        errors_ds[err_diff_var] = errors_ds[err_var_geos] - errors_ds[var]
        
        errors_ds[err_diff_rel_var] = (
            errors_ds[err_diff_var] / np.abs(errors_ds[err_var_geos]) * 100
        )
        
    return errors_ds


def compute_eke_binned_errors(
    errors_df: pd.DataFrame, 
    eke_drifter_observations: np.ndarray, 
    eke_quantiles: np.ndarray
) -> xr.Dataset:
    binning = pyinterp.Binning1D(pyinterp.Axis(eke_quantiles))
    data_vars = {}

    binning.push(eke_drifter_observations, errors_df["err_Geostrophy"])
    data_vars["err_mean_Geostrophy"] = (
        ["eke"],
        binning.variable("mean").ravel(),
        {"units": "$m/s$"}
    )
    data_vars["err_sd_Geostrophy"] = (
        ["eke"],
        np.sqrt(binning.variable("variance")).ravel(),
        {"units": "$m/s$"}
    )

    binning.clear()
    binning.push(eke_drifter_observations, errors_df["err_Cyclogeostrophy"])
    data_vars["err_mean_Cyclogeostrophy"] = (
        ["eke"],
        binning.variable("mean").ravel(),
        {"units": "$m/s$"}
    )
    data_vars["err_sd_Cyclogeostrophy"] = (
        ["eke"],
        np.sqrt(binning.variable("variance")).ravel(),
        {"units": "$m/s$"}
    )

    data_vars["count"] = (
        ["eke"],
        binning.variable("count").ravel(),
        {}
    )

    coords = {
        "eke": (["eke"], eke_quantiles)
    }
    return xr.Dataset(data_vars=data_vars, coords=coords)
