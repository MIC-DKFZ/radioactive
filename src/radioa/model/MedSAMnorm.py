from pathlib import Path
from loguru import logger
import torch
import numpy as np
import torch.nn.functional as F
import cv2
import nibabel as nib


from radioa.model.MedSAM import MedSAMInferer
from radioa.model.inferer import Inferer
from radioa.prompts.prompt import PromptStep
from radioa.utils.MedSAM_segment_anything import sam_model_registry as registry_medsam
from radioa.utils.transforms import orig_to_SAR_dense, orig_to_canonical_sparse_coords
from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib


def load_medsam(checkpoint_path, device="cuda"):
    medsam_model = registry_medsam["vit_b"](checkpoint=checkpoint_path)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    return medsam_model


class MedSAMNormInferer(MedSAMInferer):
    def preprocess_img(self, img, slices_to_process):
        slices_processed = {}
        for slice_idx in slices_to_process:
            slice = img[slice_idx, ...]
            lower_bound, upper_bound = np.percentile(slice[slice > 0], 0.5), np.percentile(slice[slice > 0], 99.5)
            slice = np.clip(slice, lower_bound, upper_bound)

            slice = np.repeat(
                slice[..., np.newaxis], repeats=3, axis=2
            )  # Repeat three times along a new final axis to simulate being a color image.

            slice = cv2.resize(slice, (1024, 1024), interpolation=cv2.INTER_CUBIC)

            slice = (slice - slice.min()) / np.clip(
                slice.max() - slice.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1]
            slice = slice.transpose(2, 0, 1)[None]  # HWC -> NCHW

            slices_processed[slice_idx] = torch.from_numpy(slice).float()

        return slices_processed
