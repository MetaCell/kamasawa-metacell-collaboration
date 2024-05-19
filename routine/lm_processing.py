import cv2
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import zscore
from skimage.filters import threshold_local
from skimage.measure import find_contours, label, regionprops_table
from skimage.morphology import convex_hull_image, dilation, skeletonize


def bin_otsu(im: np.ndarray, th_low=0, th_high=255):
    return cv2.threshold(im, th_low, th_high, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def bin_zthres(im: np.ndarray, zthres):
    return zscore(im.astype(float)) > zthres


def bin_adapt(im: np.ndarray, **kwargs):
    return cv2.adaptiveThreshold(
        im, maxValue=1, thresholdType=cv2.THRESH_BINARY, **kwargs
    )


def threshold_zstack(
    im: xr.DataArray,
    clip=(0, 255),
    nkeep=1,
    neg_space=False,
    keep_by="area",
    props=["area"],
    **kwargs
):
    im_thres = xr.apply_ufunc(
        threshold_local,
        im.clip(clip[0], clip[1]),
        input_core_dims=[["z", "y", "x"]],
        output_core_dims=[["z", "y", "x"]],
        kwargs=kwargs,
    )
    if neg_space:
        im_bin = im < im_thres
    else:
        im_bin = im > im_thres
    im_lab = label(im_bin)
    prop_df = pd.DataFrame(regionprops_table(im_lab, im, ["label"] + props))
    if nkeep is not None and keep_by is not None:
        lab_keep = prop_df.sort_values(keep_by, ascending=False)["label"].iloc[:nkeep]
        if neg_space:
            im_bin = np.full_like(im_bin, 1)
        else:
            im_bin = np.zeros_like(im_bin)
        for lab in lab_keep:
            if neg_space:
                im_bin[im_lab == lab] = 0
            else:
                im_bin[im_lab == lab] = 1
    im_bin = xr.DataArray(im_bin, dims=im.dims, coords=im.coords)
    return im_bin, prop_df


def im_convex(im: xr.DataArray):
    return xr.apply_ufunc(
        convex_hull_image,
        im,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,
    )


def prompt_pos(im: xr.DataArray):
    return xr.apply_ufunc(
        skeletonize,
        im.astype(np.uint8),
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,
    )


def prompt_neg(im: xr.DataArray, dl_wnd: tuple):
    kn = np.ones(dl_wnd)
    im_dl = xr.apply_ufunc(
        dilation,
        im,
        input_core_dims=[["z", "y", "x"]],
        output_core_dims=[["z", "y", "x"]],
        kwargs={"footprint": kn},
    )
    im_cnt = xr.apply_ufunc(
        contours,
        im_dl,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,
    )
    return im_cnt


def contours(im: np.ndarray):
    cnts = find_contours(im)
    cnt_im = np.zeros_like(im, dtype=bool)
    for cnt in cnts:
        c = np.around(cnt).astype(int)
        cnt_im[c[:, 0], c[:, 1]] = 1
    return cnt_im
