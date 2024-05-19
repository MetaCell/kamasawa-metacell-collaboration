# %% import and definition
import os

import numpy as np
import plotly.express as px
import xarray as xr

from routine.io import load_tif
from routine.lm_processing import im_convex, prompt_neg, prompt_pos, threshold_zstack

IN_LM_IM = "./data/DFJ05_ROI2_crop_LM2x_zelongation_90slices.tif"
OUT_PATH = "./output/lm_label"
FIG_PATH = "./figs/lm_process"
PARAM_CLIP = (105, 225)
PARAM_THRES_BLOCK = (3, 301, 301)
PARAM_THRES_OFFSET = 10
PARAM_DL_WND = (3, 50, 50)
PARAM_NSAMP = 100

os.makedirs(FIG_PATH, exist_ok=True)
os.makedirs(OUT_PATH, exist_ok=True)

# %% load and threshold
lm = load_tif(IN_LM_IM)
lm_bin, prop_df = threshold_zstack(
    lm,
    clip=PARAM_CLIP,
    block_size=PARAM_THRES_BLOCK,
    method="mean",
    offset=PARAM_THRES_OFFSET,
)

# %% generate prompts
lm_cvx = im_convex(lm_bin)
lm_pos = prompt_pos(lm_cvx)
lm_neg = prompt_neg(lm_cvx, PARAM_DL_WND)

# %% plot prompt result
lm_prompt = (lm_bin * 128 + lm_pos * 255 + lm_neg * 255).clip(0, 255)
lm_plt = (
    xr.concat(
        [lm.assign_coords(typ="raw"), lm_prompt.assign_coords(typ="prompt")], dim="typ"
    )
    .astype(np.uint8)
    .coarsen({"x": 2, "y": 2})
    .max()
)
fig = px.imshow(
    lm_plt.transpose("y", "x", "z", "typ"),
    animation_frame="z",
    facet_col="typ",
    aspect="equal",
    binary_string=True,
)
fig.write_html(os.path.join(FIG_PATH, "thresholding.html"))

# %% export result
pos_df = lm_pos.to_dataframe(name="positive")
pos_df = pos_df[pos_df["positive"]].reset_index()
neg_df = lm_neg.to_dataframe(name="negative")
neg_df = neg_df[neg_df["negative"]].reset_index()
pos_df.groupby("z").sample(PARAM_NSAMP, replace=True).drop_duplicates().to_csv(
    os.path.join(OUT_PATH, "pos.csv"), index=False
)
neg_df.groupby("z").sample(PARAM_NSAMP, replace=True).drop_duplicates().to_csv(
    os.path.join(OUT_PATH, "neg.csv"), index=False
)
