from typing import Dict, Tuple

import jax
from jaxparrow.tools.kinematics import kinetic_energy, magnitude, normalized_relative_vorticity
from jaxtyping import Array, Float
import numpy as np

from ..evaluation.interpolation import interpolate_grid


def compute_kinematics(
    uv_fields: dict,
    uva_fields: dict,
    lat_u: Float[Array, "lat lon"],
    lon_u: Float[Array, "lat lon"],
    lat_v: Float[Array, "lat lon"],
    lon_v: Float[Array, "lat lon"],
    mask: Float[Array, "time lat lon"]
) -> Dict[str, Tuple[np.ndarray, Dict[str, str]]]:
    def nrv(u, v, m):
        return normalized_relative_vorticity(u, v, lat_u, lat_v, lon_u, lon_v, m)

    vmap_ke = jax.vmap(kinetic_energy, in_axes=(0, 0, 0))
    vmap_magn = jax.vmap(magnitude, in_axes=(0, 0))
    vmap_nrv = jax.vmap(nrv, in_axes=(0, 0, 0))

    kinematics_vars = {}
    for method in uv_fields.keys():
        uv = uv_fields[method]
        kinematics_vars[f"u_{method}"] = (
            interpolate_grid(uv[0], mask, axis=1, padding="left"),
            {"method": method, "what": "$u$", "units": "$m/s$"}
        )
        kinematics_vars[f"v_{method}"] = (
            interpolate_grid(uv[1], mask, axis=0, padding="left"),
            {"method": method, "what": "$v$", "units": "$m/s$"}
        )
        kinematics_vars[f"magn_{method}"] = (
            np.array(vmap_magn(*uv), dtype=np.float32),
            {"method": method, "what": "$\\| \\mathbf{u} \\|$", "units": "$m/s$"}
        )
        kinematics_vars[f"nrv_{method}"] = (
            np.array(vmap_nrv(*uv, mask), dtype=np.float32),
            {"method": method, "what": "$\\xi/f$", "units": ""}
        )

        kinematics_vars[f"eke_{method}"] = (
            np.array(vmap_ke(*uva_fields[method], mask), dtype=np.float32),
            {"method": method, "what": "$\\text{EKE}$", "units": "$(m/s)^2$"}
        )

    return kinematics_vars
