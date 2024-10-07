from hydra_zen import make_custom_builds_fn

from . import cyclogeostrophy


__all__ = ["cyclogeostrophy_conf"]


pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

cyclogeostrophy_conf = pbuilds(cyclogeostrophy.cyclogeostrophy)
