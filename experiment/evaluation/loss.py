from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jaxparrow.tools.kinematics import cyclogeostrophic_imbalance
from jaxtyping import Array, Float, Scalar

from ..evaluation.interpolation import interpolate_grid


def compute_loss_value_and_grad(
    uv_fields: dict,
    dx_u: Float[Array, "lat lon"],
    dx_v: Float[Array, "lat lon"],
    dy_u: Float[Array, "lat lon"],
    dy_v: Float[Array, "lat lon"],
    coriolis_factor_u: Float[Array, "lat lon"],
    coriolis_factor_v: Float[Array, "lat lon"],
    mask: Float[Array, "time lat lon"]
) -> Dict[str, Tuple[jnp.ndarray, Dict[str, str]]]:
    def cyclo_loss_value_and_grad(
        u_geos_u: Float[Array, "lat lon"], v_geos_v: Float[Array, "lat lon"], 
        u_cyclo_u: Float[Array, "lat lon"], v_cyclo_v: Float[Array, "lat lon"], 
        _mask: Float[Array, "lat lon"]
    ) -> Tuple[
        Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]], Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]]
    ]:
        def cyclo_loss(
                _uv_cyclo: Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]]
        ) -> Tuple[Float[Scalar, ""], Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]]]:
            _u_cyclo_u, _v_cyclo_v = _uv_cyclo

            _imbalance_u, _imbalance_v = cyclogeostrophic_imbalance(
                u_geos_u, v_geos_v, _u_cyclo_u, _v_cyclo_v,
                dx_u, dx_v, dy_u, dy_v,
                coriolis_factor_u, coriolis_factor_v,
                _mask
            )

            J_u = jnp.nansum(_imbalance_u ** 2)
            J_v = jnp.nansum(_imbalance_v ** 2)

            return J_u + J_v, (_imbalance_u, _imbalance_v)
        
        (_, (_imbalance_u, _imbalance_v)), (_grad_J_u, _grad_J_v) = jax.value_and_grad(
            cyclo_loss, has_aux=True
        )((u_cyclo_u, v_cyclo_v))

        return (_imbalance_u, _imbalance_v), (_grad_J_u, _grad_J_v)

    uv_geos = uv_fields["Geostrophy"]

    vmap_cyclo_loss_value_and_grad = jax.vmap(cyclo_loss_value_and_grad, in_axes=0)

    loss_value_and_grad_vars = {}
    for method, uv_meth in uv_fields.items():
        (imbalance_u, imbalance_v), (grad_J_u, grad_J_v) = vmap_cyclo_loss_value_and_grad(*uv_geos, *uv_meth, mask)

        imbalance_u = interpolate_grid(imbalance_u, mask, axis=1, padding="left")
        imbalance_v = interpolate_grid(imbalance_v, mask, axis=0, padding="left")
        grad_J_u = interpolate_grid(grad_J_u, mask, axis=1, padding="left")
        grad_J_v = interpolate_grid(grad_J_v, mask, axis=0, padding="left")

        loss_value_and_grad_vars[f"imbalance_u_{method}"] = (
            imbalance_u,
            {"method": method, "what": "$\\sqrt{J_u}$", "units": "$m/s$"}
        )
        loss_value_and_grad_vars[f"imbalance_v_{method}"] = (
            imbalance_v,
            {"method": method, "what": "$\\sqrt{J_v}$", "units": "$m/s$"}
        )
        loss_value_and_grad_vars[f"imbalance_uv_{method}"] = (
            (imbalance_u ** 2 + imbalance_v ** 2) ** (1 / 2),
            {"method": method, "what": "$\\sqrt{J}$", "units": "$m/s$"}
        )
        loss_value_and_grad_vars[f"grad_J_u_{method}"] = (
            grad_J_u,
            {"method": method, "what": "$\\nabla J_u$", "units": "$m/s$"}
        )
        loss_value_and_grad_vars[f"grad_J_v_{method}"] = (
            grad_J_v,
            {"method": method, "what": "$\\nabla J_v$", "units": "$m/s$"}
        )
        loss_value_and_grad_vars[f"grad_J_uv_{method}"] = (
            (grad_J_u ** 2 + grad_J_v ** 2) ** (1 / 2),
            {"method": method, "what": "$\\nabla J$", "units": "$m/s$"}
        )

    return loss_value_and_grad_vars
