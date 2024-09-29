from pathlib import Path
from pathlib import Path
from loguru import logger
import torch
import numpy as np
import torch.nn.functional as F
from typing import TypeVar
import torchio as tio
from itertools import product
import nibabel as nib


from intrab.model.SAMMed3D import SAMMed3DInferer
from intrab.model.inferer import Inferer
from intrab.prompts.prompt import PromptStep
from intrab.utils.SAMMed3D_segment_anything.build_sam3D import build_sam3D_vit_b_ori
from intrab.utils.SAMMed3D_segment_anything.modeling.sam3D import Sam3D
from intrab.utils.image import get_crop_pad_params_from_gt_or_prompt
from intrab.utils.resample import get_current_spacing_from_affine, resample_to_shape, resample_to_spacing
from intrab.utils.transforms import resample_to_spacing_sparse
from intrab.datasets_preprocessing.conversion_utils import load_any_to_nib




class SAMMed3DNormInferer(SAMMed3DInferer):
    def set_image(self, img_path: str | Path) -> None:
        if self._image_already_loaded(img_path=img_path):
            return
        img_nib = load_any_to_nib(img_path)
        self.orig_affine = img_nib.affine
        self.orig_shape = img_nib.shape

        self.img, self.inv_trans_dense = self.transform_to_model_coords_dense(img_path, is_seg=False)
        self.new_shape = self.img.shape
        self.loaded_image = img_path

        # clip image to 0.5% - 99.5%
        self.global_min = np.percentile(self.img, 0.5)
        self.global_max = np.percentile(self.img, 99.5)

        # Clip the image array
        self.img = np.clip(self.img, self.global_min, self.global_max)
