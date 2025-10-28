import numpy as np
import pandas as pd
import pyinterp
import pyinterp.backends.xarray
import xarray as xr


def gridded_eke_to_drifters_observations(eke_da: xr.DataArray, df: pd.DataFrame) -> np.ndarray:
    interpolator = pyinterp.backends.xarray.Grid3D(eke_da)
    return interpolator.trivariate({"longitude": df.lon, "latitude": df.lat, "time": np.asarray(df.time)})
