from abc import ABC, abstractmethod
from pathlib import Path
from loguru import logger
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image
import nibabel as nib
from copy import deepcopy
from typing import Callable, Tuple
from nibabel.orientations import io_orientation, ornt_transform

from intrab.prompts.prompt import Boxes3D, PromptStep
from intrab.utils.nnunet.default_resampling import compute_new_shape
from intrab.utils.resample import get_current_spacing_from_affine
from intrab.datasets_preprocessing.conversion_utils import load_any_to_nib

# The function below is from the original SAM repository

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1], self.target_length)
        coords = deepcopy(coords).astype(float)
        coords[..., 2] = coords[..., 2] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(2, 3), original_size)
        boxes = boxes[:, 1:]  # Remove z coord
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(image, target_size, mode="bilinear", align_corners=False, antialias=True)

    def apply_coords_torch(self, coords: torch.Tensor, original_size: Tuple[int, ...]) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1], self.target_length)
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(self, boxes: torch.Tensor, original_size: Tuple[int, ...]) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


def orig_to_SAR_dense(nifti: Path | nib.Nifti1Image) -> tuple[np.ndarray, Callable[[np.ndarray], nib.Nifti1Image]]:
    """ """

    if isinstance(nifti, (str, Path)):
        nifti: nib.Nifti1Image = load_any_to_nib(nifti)
    orientation_old = io_orientation(nifti.affine)

    if nib.aff2axcodes(nifti.affine) != ("R", "A", "S"):
        nifti = nib.as_closest_canonical(nifti)
    # Get orientation in canonical space
    orientation_new = io_orientation(nifti.affine)
    # Create transform back to original image space given the two orientations.
    orientation_transform = ornt_transform(orientation_new, orientation_old)
    # Get the data and re-orient to
    data = nifti.get_fdata()
    data = data.transpose(2, 1, 0)  # Reorient to zyx

    def inv_trans(arr: np.ndarray):
        # Make z y x - x y z again
        arr = arr.transpose(2, 1, 0)
        # Create image with original RAS affine (xyz affine)
        arr_nib = nib.Nifti1Image(arr, nifti.affine, nifti.header)
        # Then revert to the actual original spacing we had before any transformations.
        arr_orig_ori = arr_nib.as_reoriented(orientation_transform)
        return arr_orig_ori

    # Return the data in the new format and transformation function
    return data, inv_trans


def orig_to_canonical_sparse_coords(coords: np.ndarray, orig_affine: np.ndarray, orig_shape: tuple) -> np.ndarray:
    """
    Transform an nx3 array of coordinates aligned to an image with affine `affine` into canonical orientation.
    """
    coords = coords.copy()
    was_1d = coords.ndim == 1
    if was_1d:
        coords = coords[None]

    io_orient = io_orientation(orig_affine)
    # Reverse the axes in line with as_closest_canonical
    axes_reverse = io_orient[:, 1].astype(int)
    for i in range(3):
        if axes_reverse[i] == -1:
            coords[:, i] = orig_shape[i] - coords[:, i] - 1

    # Transpose the axes in line with as_closest_canonical
    axes_transpose = io_orient[:, 0].astype(int)
    coords = coords[:, axes_transpose]

    if was_1d:
        coords = coords[0]

    coords = coords[::-1]  # Not sure why this was necessary, but it seemed to be.

    return coords


def canonical_to_orig_sparse_coords(coords: np.ndarray, orig_affine: np.ndarray, orig_shape: tuple) -> np.ndarray:
    """
    Transform an nx3 array of coordinates in canonical form to align to an image with affine `afffine`.
    Note that this is just the two transforms in orig_to_canonical_sparse_coords in reverse order, since the
    transforms are their own inverse (but do not commute)
    original_shape/affine refers to the shape/affine in the original system, not in the canonical system
    """
    coords = coords.copy()
    was_1d = coords.ndim == 1
    if was_1d:
        coords = coords[None]

    io_orient = io_orientation(orig_affine)

    # Transpose the axes in line with as_closest_canonical
    axes_transpose = io_orient[:, 0].astype(int)
    coords = coords[:, axes_transpose]

    # Reverse the axes in line with as_closest_canonical
    axes_reverse = io_orient[:, 1].astype(int)
    for i in range(3):
        if axes_reverse[i] == -1:
            coords[:, i] = orig_shape[i] - coords[:, i] - 1

    if was_1d:
        coords = coords[0]
    return coords


def resample_to_shape_sparse(
    coords: np.ndarray, current_shape: tuple[int, int, int], target_shape: tuple[int, int, int], round=False
) -> np.ndarray:
    """
    Transform an nx2 or nx3 array of coordinates in line with resampling from an original shape to a target shape.
    Rounding to integer coordinates is supported, but if this transform is composed with others, it may introduce rounding issues.
    """
    coords = coords * target_shape / current_shape
    if round:
        coords = np.round(coords)

    return coords


def resample_to_spacing_sparse(
    coords: np.ndarray,
    current_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    current_shape: tuple[int, int, int],
    round=False,
) -> np.ndarray:
    """
    Transform an nx3 array of coordsinates in line with a resampling from an original spacing to a new spacing.
    Rounding to integer coordinates is supported, but if this transform is composed with others, it may introduce rounding issues.
    Warning: compute_new_shape can introduce rounding errors, so this function is not a true self-inverse
    """
    target_shape = compute_new_shape(current_shape, current_spacing, target_spacing)
    coords = resample_to_shape_sparse(coords, current_shape, target_shape, round)

    return coords


def _transform_boxes3d_to_model_coords(box: Boxes3D, transform_coords: Callable[[np.ndarray], np.ndarray]) -> Boxes3D:
    """
    Takes a boxes3d prompt in the original coordinate system and transforms it into the model system
    """
    min_vertex, max_vertex = box.bbox
    vertices_combined = np.array([transform_coords(min_vertex), transform_coords(max_vertex)])

    min_vertex_transformed = np.max(vertices_combined, axis=0)
    max_vertex_transformed = np.min(vertices_combined, axis=0)

    box_model = Boxes3D(min_vertex_transformed, max_vertex_transformed)

    return box_model


def _transform_box_dict_to_model_coords(
    box_dict: dict[int, np.ndarray],
    transform_coords: Callable[[np.ndarray], np.ndarray],
    transform_reverses_order: bool,
) -> dict[int, np.ndarray]:

    box_dict_transformed = {}
    for z, xyxy in box_dict.items():
        min_vertex = np.array((*xyxy[:2], z))
        max_vertex = np.array((*xyxy[2:], z))

        vertices_combined = np.array([transform_coords(min_vertex), transform_coords(max_vertex)])
        if transform_reverses_order:
            vertices_combined = vertices_combined[:, ::-1]

        min_vertex_transformed = np.min(vertices_combined, axis=0)
        max_vertex_transformed = np.max(vertices_combined, axis=0)

        if min_vertex_transformed[2] != max_vertex_transformed[2]:
            logger.warning(
                "Transformed bounding box does not lie in one slice in the final axis; was it formatted correctly on input?"
            )

        x_min, y_min, _ = min_vertex_transformed
        x_max, y_max, z_new = max_vertex_transformed

        box_dict_transformed[z_new] = np.array([x_min, y_min, x_max, y_max])

    return box_dict_transformed


def _transform_points_prompt_to_model_coords(
    coords_orig: np.ndarray, labels_orig: np.ndarray, transform_coords: Callable[[np.ndarray], np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform a point prompt in the original coordinate system to the model coordinate system using transform_to_model_coords_sparse
    """
    coords_transformed = []
    for coord_triple in coords_orig:
        coords_transformed.append(transform_coords(coord_triple))

    coords_transformed = np.array(coords_transformed)

    return coords_transformed, labels_orig


def transform_prompt_to_model_coords(
    prompt_orig: PromptStep | Boxes3D,
    transform_coords: Callable[[np.ndarray], np.ndarray],
    transform_reverses_order: bool,
):
    # Deal with special case: Handle 3D boxes
    if isinstance(prompt_orig, Boxes3D):
        return _transform_boxes3d_to_model_coords(prompt_orig)

    # Initialise empty promptstep
    prompt_model = PromptStep()

    # set points if needed
    if prompt_orig.has_points:
        coords_model, labels_model = _transform_points_prompt_to_model_coords(
            prompt_orig.coords, prompt_orig.labels, transform_coords
        )
        prompt_model.set_points((coords_model.astype(int), labels_model))

    # set boxes if needed
    if prompt_orig.has_boxes:
        box_dict_model = _transform_box_dict_to_model_coords(
            prompt_orig.boxes, transform_coords, transform_reverses_order
        )
        prompt_model.set_boxes(box_dict_model)

    # Set masks - do not transform, mask prompts should always remain in model coordinates
    if prompt_orig.has_masks:
        prompt_model.set_masks(prompt_orig.masks)

    return prompt_model
