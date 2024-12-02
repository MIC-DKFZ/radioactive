from pathlib import Path

import numpy as np
import torch
from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib
from radioa.model.SAM import SAMInferer
import nibabel as nib
import torch.nn.functional as F


class SAMNormInferer(SAMInferer):

    def set_image(self, img_path: Path | str):
        img_path = Path(img_path)
        if self._image_already_loaded(img_path=img_path):
            return
        img_nib = load_any_to_nib(img_path)
        self.orig_affine = img_nib.affine
        self.orig_shape = img_nib.shape

        self.img, self.inv_trans_dense = self.transform_to_model_coords_dense(img_nib, is_seg=False)
        self.loaded_image = img_path
        self.new_shape = self.img.shape
        self.loaded_image = img_path

        # clip image to 0.5% - 99.5%
        self.global_min = np.percentile(self.img, 0.5)
        self.global_max = np.percentile(self.img, 99.5)

        # Clip the image array
        self.img = np.clip(self.img, self.global_min, self.global_max)

    def preprocess_img(self, img, slices_to_process):
        """
        Preprocessing steps
            - Extract slice, resize (maintaining aspect ratio), pad edges
        """

        # Perform slicewise processing and collect back into a volume at the end
        slices_processed = {}
        for slice_idx in slices_to_process:
            slice = img[slice_idx, ...]  # Now HW
            slice = np.round((slice - self.global_min) / (self.global_max - self.global_min + 1e-10) * 255.0).astype(
                np.uint8
            )  # Change to 0-255 scale
            slice = np.repeat(slice[..., None], repeats=3, axis=-1)  # Add channel dimension to make it RGB-like
            slice = self.transform.apply_image(slice)
            slice = torch.as_tensor(slice, device=self.device)
            slice = slice.permute(2, 0, 1).contiguous()[
                None, :, :, :
            ]  # Change to BCHW, make memory storage contiguous.

            if self.input_size is None:
                self.input_size = tuple(
                    slice.shape[-2:]
                )  # Store the input size pre-padding if it hasn't been done yet

            slice = (slice - self.pixel_mean) / self.pixel_std

            h, w = slice.shape[-2:]
            padh = self.model.image_encoder.img_size - h
            padw = self.model.image_encoder.img_size - w
            slice = F.pad(slice, (0, padw, 0, padh))

            slices_processed[slice_idx] = slice
        self.slices_processed = slices_processed
        return slices_processed
