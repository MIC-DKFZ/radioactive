from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Sequence

from loguru import logger
import nibabel as nib
import numpy as np
from intrab.prompts.prompt import Boxes3D, PromptStep
from intrab.utils.transforms import transform_prompt_to_model_coords

# Mask corresponds to being an interactive method allowing
prompt_categories = Literal["box", "point", "mask"]


class Inferer(ABC):
    supported_prompts: Sequence[prompt_categories] = ...
    dim: Literal[2, 3] = ...
    pass_prev_prompts: bool = ...

    def __init__(self, checkpoint_path, device):
        self.device = device
        self.model = self.load_model(checkpoint_path, device)
        self.loaded_image: Path | None = None
        # Just used for 2D models and not for 3D models
        self.image_embeddings_dict = {}
        
        self.orig_affine: tuple[float, float, float] = None
        self.orig_shape: tuple[int, int, int] = None
        self.img: np.ndarray = None
        self.new_shape: tuple[int, int, int] = None
        self.inv_trans_dense: Callable[[np.ndarray], nib.Nifti1Image] = None

    @abstractmethod
    def load_model(self, checkpoint_path, device):
        """Load the model from a checkpoint"""
        pass

    @abstractmethod
    def preprocess_img(self, img: np.ndarray):
        """Any necessary preprocessing step like"""
        pass

    @abstractmethod
    def preprocess_prompt(self, prompt: PromptStep):
        pass

    # ToDo: Check if this should be an abstract method.
    #   Currently MedSAM did not have this.
    def postprocess_mask(self, mask):
        """Any necessary postprocessing steps"""
        pass

    @abstractmethod
    def predict(
        self, prompts: PromptStep, prev_seg: nib.Nifti1Image, promptstep_in_model_coord_system: bool,
    ) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
        """Obtain logits"""
        pass

    def _image_already_loaded(self, img_path):
        return self.loaded_image is not None and img_path == self.loaded_image

    @abstractmethod
    def set_image(self, img_path):
        """Set the image to be segmented and make sure it adheres to local model requirements"""
        pass

    @abstractmethod
    def transform_to_model_coords_dense(self, nifti: Path | nib.Nifti1Image, is_seg: bool) -> np.ndarray:
        """Transform a dense array (segmentation or image) to the model's coordinate system"""
        pass

    @abstractmethod
    def transform_to_model_coords_sparse(self, coords: np.ndarray) -> np.ndarray:
        """Transform a sparse array (sparse coordinates) to the model's coordinate system"""
        pass

    def get_transformed_groundtruth(self, nifti: Path | nib.Nifti1Image) -> np.ndarray:
        """Transforms the nifti or the groundtruth to the model's coordinate system."""
        return self.transform_to_model_coords_dense(nifti, is_seg=True)[0]
    
    def transform_promptstep_to_model_coords(self, prompt_orig: PromptStep) -> PromptStep:
        return transform_prompt_to_model_coords(prompt_orig, self.transform_to_model_coords_sparse)

    def merge_seg_with_prev_seg(
        self, new_seg: np.ndarray, prev_seg: str | Path | nib.Nifti1Image, slices_inferred: np.ndarray
    ):
        # Find slices which were inferred on in old seg, but not in new_seg
        prev_seg, _ = self.transform_to_model_coords_dense(prev_seg, None)
        old_seg_inferred_slices = np.where(np.any(prev_seg, axis=(1, 2)))[0]
        missing_slices = np.setdiff1d(old_seg_inferred_slices, slices_inferred)

        # Merge segmentations
        new_seg[missing_slices] = prev_seg[missing_slices]

        return new_seg
