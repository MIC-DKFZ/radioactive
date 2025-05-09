from typing import Tuple
import numpy as np
from radioa.utils.nnunet.resample_torch import resample_torch_fornnunet
from radioa.utils.nnunet.default_resampling import compute_new_shape


def get_current_spacing_from_affine(affine: np.ndarray):
    return np.abs(affine.diagonal()[:-1])


def resample_to_spacing(
    seg: np.ndarray,
    current_spacing: Tuple | np.ndarray,
    new_spacing: Tuple | np.ndarray = (1.5, 1.5, 1.5),
    is_seg: bool = True,
):
    """
    Resample an image to a given spacing.
    Warning: This function is not a true self inverse. In particular, the dense version applied twice to a mask might not
    return something of the same shape. Thus, when returning to the original spacing, it's better to use resample_to_shape_sparse
    using the original shape.
    """
    # Find new shape to change to
    new_shape = compute_new_shape(seg.shape, current_spacing, new_spacing)

    seg_resampled = resample_to_shape(seg, current_spacing, new_shape, new_spacing, is_seg)
    return seg_resampled


def resample_to_shape(
    seg: np.ndarray,
    current_spacing: Tuple | np.ndarray,
    new_shape,
    new_spacing: Tuple | np.ndarray = (1.5, 1.5, 1.5),
    is_seg: bool = True,
):
    # Change gt to cxyz
    seg = seg[None]
    # Perform resampling
    seg_resampled = resample_torch_fornnunet(
        seg, new_shape, current_spacing, new_spacing, is_seg=is_seg, device="cuda", memefficient_seg_resampling=True
    )

    seg_resampled = seg_resampled[0]

    return seg_resampled
