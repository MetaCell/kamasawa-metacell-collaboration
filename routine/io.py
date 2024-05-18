import numpy as np
import xarray as xr
from tifffile import imread


def load_tif(lm_imag):
    arr = xr.DataArray(imread(lm_imag), dims=["z", "y", "x"])
    arr = arr.assign_coords(
        {
            "x": np.arange(arr.sizes["x"]),
            "y": np.arange(arr.sizes["y"]),
            "z": np.arange(arr.sizes["z"]),
        }
    )
    return arr
