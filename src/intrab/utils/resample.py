from typing import Tuple
import numpy as np
from intrab.utils.nnunet.resample_torch import resample_torch_fornnunet
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape

def get_current_spacing_from_affine(affine: np.ndarray):
    return np.abs(affine.diagonal()[:-1])

def resample(seg:np.ndarray, current_spacing: Tuple|np.ndarray, new_spacing: Tuple|np.ndarray = (1.5, 1.5, 1.5), is_seg:bool = True):
    # Find new shape to change to
    new_shape = compute_new_shape(seg.shape, current_spacing, new_spacing)
    # Change gt to cxyz
    seg = seg[None]
    # Perform resampling
    seg_resampled = resample_torch_fornnunet(
        seg,
        new_shape,
        current_spacing,
        new_spacing, 
        is_seg = is_seg,
        device = 'cuda', memefficient_seg_resampling=True
    )

    seg_resampled = seg_resampled[0]

    return(seg_resampled)