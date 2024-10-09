from functools import partial
import math
from typing import Callable, List, Tuple

import numpy as np
import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float


def pad(n_batches: int, arrs: List[Float[Array, "time lat lon"]]) -> Tuple[List[Float[Array, "time_pad lat lon"]], int]:
    batch_dim = arrs[0].shape[0]
    div_res = math.ceil(batch_dim / n_batches)
    pad_with = n_batches * div_res - batch_dim
    if pad_with > 0:
        for i in range(len(arrs)):
            arrs[i] = jnp.pad(arrs[i], ((0, pad_with), (0, 0), (0, 0)))
    return arrs, pad_with


def unpad(pad_with: int, arrs: List[Float[Array, "time lat lon"]]) -> List[Float[Array, "(time) lat lon"]]:
    if pad_with > 0:
        for i in range(len(arrs)):
            if arrs[i].shape == 3:  # unpad does not apply to lat lon arrays
                arrs[i] = arrs[i][:-pad_with, :, :]
    return arrs


def pad_wrapper(
    n_batches: int,
    f: Callable,
    arrs: List[Float[Array, "time lat lon"]]
) -> List[Float[Array, "time lat lon"]]:
    pad_arrs, pad_with = pad(n_batches, arrs)
    pad_res = f(*pad_arrs)
    unpad_res = unpad(pad_with, pad_res)
    return unpad_res


def parallelize_and_vectorize(f: Callable) -> Callable:
    # vectorized fn
    vmap_f = jax.vmap(f, in_axes=(0, 0), out_axes=(0, 0, 0, 0, None, None, None, None))

    # parallelized fn
    devices = np.array(jax.devices())
    n_devices = len(devices)
    mesh = Mesh(devices, axis_names=("time",))
    out_specs = (
        P("time", None, None), P("time", None, None),
        P("time", None, None), P("time", None, None),
        P(None, None), P(None, None),
        P(None, None), P(None, None)
    )
    shmap_f = shard_map(
        vmap_f,
        mesh=mesh,
        in_specs=(P("time", None, None), P("time", None, None)),
        out_specs=out_specs,
        check_rep=False
    )

    return partial(pad_wrapper, n_devices, shmap_f)  # add a wrapper to handle padding
