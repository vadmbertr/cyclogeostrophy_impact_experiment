from typing import Literal, List

import jaxparrow as jpw
from jaxtyping import Array, Float

from ._map import parallelize_and_vectorize


def cyclogeostrophy(
    lat_t: Float[Array, "lat lon"], lon_t: Float[Array, "lat lon"],
    adt_t: Float[Array, "time lat lon"], mask: Float[Array, "time lat lon"],
    method: Literal["variational", "iterative"] = "variational",
    n_it: int = 2000,
    var_optim: str = "sgd",
    var_lr: float = 0.005,
    it_use_filter: bool = False,
    it_filter_size: int = 3
) -> List[Float[Array, "(time) lat lon"]]:
    # convenient wrapper to avoid too much verbosity when using vmap and shmap
    def partial_cyclogeostrophy(partial_adt_t, partial_mask):
        return jpw.cyclogeostrophy(
            partial_adt_t, lat_t, lon_t, partial_mask,
            method=method, n_it=n_it, optim=var_optim, optim_kwargs={"learning_rate": var_lr},
            use_res_filter=it_use_filter, res_filter_size=it_filter_size,
            return_geos=True
        )
    # parallelize and vectorize over the time dimension
    pv_cyclo_fn = parallelize_and_vectorize(partial_cyclogeostrophy)

    return pv_cyclo_fn([adt_t, mask])
