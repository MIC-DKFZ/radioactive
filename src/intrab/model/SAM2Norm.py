from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger
from intrab.datasets_preprocessing.conversion_utils import load_any_to_nib
from intrab.model.SAM import SAMInferer
from intrab.model.inferer import Inferer
from intrab.prompts.prompt import PromptStep
import numpy as np


class SAM2NormInferer(SAMInferer):
    def preprocess_img(self, img, slices_to_process):
        # Perform slicewise processing and collect back into a volume at the end
        slices_processed = {}
        for slice_idx in slices_to_process:
            slice = img[slice_idx, ...]  # Now HW
            slice = np.round((slice - self.global_min) / (self.global_max - self.global_min + 1e-10) * 255.0).astype(
                np.uint8
            )  # Change to 0-255 scale
            slice = np.repeat(slice[..., None], repeats=3, axis=-1)  # Add channel dimension to make it RGB-like -> now HWC

            input_slice = self._transforms(slice)
            input_slice = input_slice[None, ...].to(self.device)
            slices_processed[slice_idx] = input_slice
        return slices_processed

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
        self._orig_hw = self.img[0].shape
        self.loaded_image = img_path

        # clip image to 0.5% - 99.5%
        self.global_min = np.percentile(self.img, 0.5)
        self.global_max = np.percentile(self.img, 99.5)

        # Clip the image array
        self.img = np.clip(self.img, self.global_min, self.global_max)