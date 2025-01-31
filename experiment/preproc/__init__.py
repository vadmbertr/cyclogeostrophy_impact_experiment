from hydra_zen import store, make_custom_builds_fn
import toolz
import xarray as xr

from . import drifter, ssh


__all__ = [
    "drifter_preproc_store",
    "drifter_default_preproc_conf",
    "ssh_preproc_store",
    "ssh_lon_to_180_180_preproc_conf"
]


def no_preproc(ds: xr.Dataset) -> xr.Dataset:
    return ds


def apply_drifter_preproc(drifter_ds: xr.Dataset, preproc_steps: tuple) -> xr.Dataset:
    return toolz.compose_left(*[getattr(drifter, step) for step in preproc_steps])(drifter_ds)


def apply_ssh_preproc(ssh_ds: xr.Dataset, preproc_steps: tuple) -> xr.Dataset:
    return toolz.compose_left(*[getattr(ssh, step) for step in preproc_steps])(ssh_ds)


pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

no_preproc_conf = pbuilds(no_preproc)

drifter_default_preproc_conf = pbuilds(apply_drifter_preproc, preproc_steps=drifter.DEFAULT_STEPS)
drifter_preproc_store = store(group="drifter_preproc")
drifter_preproc_store(drifter_default_preproc_conf, name="default")
drifter_preproc_store(no_preproc_conf, name="none")

ssh_default_preproc_conf = pbuilds(apply_ssh_preproc, preproc_steps=ssh.DEFAULT_STEPS)
ssh_preproc_store = store(group="ssh_preproc")
ssh_preproc_store(ssh_default_preproc_conf, name="default")
ssh_preproc_store(no_preproc_conf, name="none")
