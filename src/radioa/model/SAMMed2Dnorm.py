from pathlib import Path
from typing import Sequence
import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from argparse import Namespace
import nibabel as nib
from loguru import logger

from radioa.model.SAMMed2D import SAMMed2DInferer
from radioa.utils.SAMMed2D_segment_anything import sam_model_registry as registry_sammed2d
from radioa.prompts.prompt import PromptStep
from radioa.model.inferer import Inferer
from radioa.utils.transforms import orig_to_SAR_dense, orig_to_canonical_sparse_coords
from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib


def load_sammed2d(checkpoint_path, image_size, device="cuda"):
    args = Namespace()
    args.image_size = image_size
    args.encoder_adapter = True
    args.sam_checkpoint = checkpoint_path
    model = registry_sammed2d["vit_b"](args).to(device)
    model.eval()

    return model


class SAMMed2DNormInferer(SAMMed2DInferer):
    def preprocess_img(self, img, slices_to_process):
        slices_processed = {}
        for slice_idx in slices_to_process:
            slice = img[slice_idx, ...]
            lower_bound, upper_bound = np.percentile(slice[slice > 0], 0.5), np.percentile(slice[slice > 0], 99.5)
            slice = np.clip(slice, lower_bound, upper_bound)

            slice = np.round((slice - slice.min()) / (slice.max() - slice.min() + 1e-6) * 255).astype(
                np.uint8
            )  # Get slice into [0,255] rgb scale
            slice = np.repeat(slice[..., None], repeats=3, axis=-1)  # Add channel dimension to make it RGB-like
            slice = (slice - self.pixel_mean) / self.pixel_std  # normalise

            transforms = self.transforms(self.new_size)
            augments = transforms(image=slice)
            slice = augments["image"][None, :, :, :]  # Add batch dimension

            slices_processed[slice_idx] = slice.float()

        return slices_processed
