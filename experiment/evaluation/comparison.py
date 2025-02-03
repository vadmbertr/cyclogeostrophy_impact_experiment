from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr


def _differences(
    ref: xr.DataArray | pd.Series, 
    other: xr.DataArray | pd.Series
) -> Tuple[xr.DataArray | pd.Series, xr.DataArray | pd.Series]:
    abs_diff = other - ref
    rel_diff = 100 * abs_diff / np.abs(other)  # .where(other > 1e-3, 0)
    return abs_diff, rel_diff


def _compare(
    errors_df: pd.DataFrame,
    kinematics_ds: xr.Dataset,
    ref_method: str,
    method: str
) -> Tuple[xr.Dataset, xr.Dataset]:
    abs_diff, rel_diff = _differences(kinematics_ds[f"u_{ref_method}"], kinematics_ds[f"u_{method}"])
    kinematics_ds[f"u_diff_{ref_method}_{method}"] = -abs_diff  # note the "-"
    kinematics_ds[f"u_diff_{ref_method}_{method}"].attrs = {
        "method": f"{ref_method}(1) ; {method}(2)", "units": "$m/s$",
        "what": "$\\langle u_1 \\rangle - \\langle u_2 \\rangle$"
    }
    kinematics_ds[f"u_diff_rel_{ref_method}_{method}"] = -rel_diff  # note the "-"
    kinematics_ds[f"u_diff_rel_{ref_method}_{method}"].attrs = {
        "method": f"{ref_method}(1) ; {method}(2)", "units": "$\\%$",
        "what": "$100 (\\langle u_1 \\rangle - \\langle u_2 \\rangle) / \\langle u_2 \\rangle$"
    }

    abs_diff, rel_diff = _differences(kinematics_ds[f"v_{ref_method}"], kinematics_ds[f"v_{method}"])
    kinematics_ds[f"v_diff_{ref_method}_{method}"] = -abs_diff  # note the "-"
    kinematics_ds[f"v_diff_{ref_method}_{method}"].attrs = {
        "method": f"{ref_method}(1) ; {method}(2)", "units": "$m/s$",
        "what": "$\\langle v_1 \\rangle - \\langle v_2 \\rangle$"
    }
    kinematics_ds[f"v_diff_rel_{ref_method}_{method}"] = -rel_diff  # note the "-"
    kinematics_ds[f"v_diff_rel_{ref_method}_{method}"].attrs = {
        "method": f"{ref_method}(1) ; {method}(2)", "units": "$\\%$",
        "what": "$100 (\\langle v_1 \\rangle - \\langle v_2 \\rangle) / \\langle v_2 \\rangle$"
    }

    abs_diff, rel_diff = _differences(kinematics_ds[f"magn_{ref_method}"], kinematics_ds[f"magn_{method}"])
    kinematics_ds[f"magn_diff_{ref_method}_{method}"] = -abs_diff  # note the "-"
    kinematics_ds[f"magn_diff_{ref_method}_{method}"].attrs = {
        "method": f"{ref_method}(1) ; {method}(2)", "units": "$m/s$",
        "what": "$\\langle \\| \\mathbf{u}_1 \\| \\rangle - \\langle \\| \\mathbf{u}_2 \\| \\rangle$"
    }
    kinematics_ds[f"magn_diff_rel_{ref_method}_{method}"] = -rel_diff  # note the "-"
    kinematics_ds[f"magn_diff_rel_{ref_method}_{method}"].attrs = {
        "method": f"{ref_method}(1) ; {method}(2)", "units": "$\\%$",
        "what": "$100 (\\langle \\| \\mathbf{u}_1 \\| \\rangle - \\langle \\| \\mathbf{u}_2 \\| \\rangle) / "
                "\\langle \\| \\mathbf{u}_2 \\| \\rangle$"
    }

    abs_diff, rel_diff = _differences(kinematics_ds[f"nrv_{ref_method}"], kinematics_ds[f"nrv_{method}"])
    kinematics_ds[f"nrv_diff_{ref_method}_{method}"] = -abs_diff  # note the "-"
    kinematics_ds[f"nrv_diff_{ref_method}_{method}"].attrs = {
        "method": f"{ref_method}(1) ; {method}(2)", "units": "",
        "what": "$\\langle \\xi_1/f \\rangle - \\langle \\xi_2/f \\rangle$"
    }
    kinematics_ds[f"nrv_diff_rel_{ref_method}_{method}"] = -rel_diff  # note the "-"
    kinematics_ds[f"nrv_diff_rel_{ref_method}_{method}"].attrs = {
        "method": f"{ref_method}(1) ; {method}(2)", "units": "$\\%$",
        "what": "$100 (\\langle \\xi_1/f \\rangle - \\langle \\xi_2/f \\rangle) / \\langle \\xi_2/f \\rangle$"
    }

    abs_diff, rel_diff = _differences(kinematics_ds[f"eke_{ref_method}"], kinematics_ds[f"eke_{method}"])
    kinematics_ds[f"eke_diff_{ref_method}_{method}"] = -abs_diff  # note the "-"
    kinematics_ds[f"eke_diff_{ref_method}_{method}"].attrs = {
        "method": f"{ref_method}(1) ; {method}(2)", "units": "$(m/s)^2$",
        "what": "$\\langle \\text{EKE}_1 \\rangle - \\langle \\text{EKE}_2 \\rangle$"
    }
    kinematics_ds[f"eke_diff_rel_{ref_method}_{method}"] = -rel_diff  # note the "-"
    kinematics_ds[f"eke_diff_rel_{ref_method}_{method}"].attrs = {
        "method": f"{ref_method}(1) ; {method}(2)", "units": "$\\%$",
        "what": "$100 (\\langle \\text{EKE}_1 \\rangle - \\langle \\text{EKE}_2 \\rangle) / "
                "\\langle \\text{EKE}_2 \\rangle$"
    }

    if errors_df is not None:
        abs_diff, rel_diff = _differences(errors_df[f"err_u_{ref_method}"], errors_df[f"err_u_{method}"])
        errors_df[f"err_u_diff_{ref_method}_{method}"] = abs_diff
        errors_df[f"err_u_diff_{ref_method}_{method}"].attrs = {
            "method": f"{ref_method}(1) ; {method}(2)", "units": "$m/s$",
            "what": "$\\langle \\epsilon_{2_u} \\rangle - \\langle \\epsilon_{1_u} \\rangle$"
        }
        errors_df[f"err_u_diff_rel_{ref_method}_{method}"] = rel_diff
        errors_df[f"err_u_diff_rel_{ref_method}_{method}"].attrs = {
            "method": f"{ref_method}(1) ; {method}(2)", "units": "$\\%$",
            "what": "$100 (\\langle \\epsilon_{2_u} \\rangle - \\langle \\epsilon_{1_u} \\rangle) / "
                    "\\langle \\epsilon_{2_u} \\rangle$"
        }

        abs_diff, rel_diff = _differences(errors_df[f"err_v_{ref_method}"], errors_df[f"err_v_{method}"])
        errors_df[f"err_v_diff_{ref_method}_{method}"] = abs_diff
        errors_df[f"err_v_diff_{ref_method}_{method}"].attrs = {
            "method": f"{ref_method}(1) ; {method}(2)", "units": "$m/s$",
            "what": "$\\langle \\epsilon_{2_v} \\rangle - \\langle \\epsilon_{1_v} \\rangle$"
        }
        errors_df[f"err_v_diff_rel_{ref_method}_{method}"] = rel_diff
        errors_df[f"err_v_diff_rel_{ref_method}_{method}"].attrs = {
            "method": f"{ref_method}(1) ; {method}(2)", "units": "$\\%$",
            "what": "$100 (\\langle \\epsilon_{2_v} \\rangle - \\langle \\epsilon_{1_v} \\rangle) / "
                    "\\langle \\epsilon_{1_v} \\rangle$"
        }

        abs_diff, rel_diff = _differences(errors_df[f"err_{ref_method}"], errors_df[f"err_{method}"])
        errors_df[f"err_diff_{ref_method}_{method}"] = abs_diff
        errors_df[f"err_diff_{ref_method}_{method}"].attrs = {
            "method": f"{ref_method}(1) ; {method}(2)", "units": "$m/s$",
            "what": "$\\langle \\epsilon_2 \\rangle - \\langle \\epsilon_1 \\rangle$"
        }
        errors_df[f"err_diff_rel_{ref_method}_{method}"] = rel_diff
        errors_df[f"err_diff_rel_{ref_method}_{method}"].attrs = {
            "method": f"{ref_method}(1) ; {method}(2)", "units": "$\\%$",
            "what": "$100 (\\langle \\epsilon_2 \\rangle - \\langle \\epsilon_1 \\rangle) / "
                    "\\langle \\epsilon_2 \\rangle$"
        }

    return errors_df, kinematics_ds


def compare_methods(errors_df: pd.DataFrame, kinematics_ds: xr.Dataset) -> Tuple[pd.DataFrame, xr.Dataset]:
    return _compare(errors_df, kinematics_ds, ref_method="Cyclogeostrophy", method="Geostrophy")
