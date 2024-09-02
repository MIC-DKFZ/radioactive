from abc import abstractmethod
from pathlib import Path
from typing import Literal
import numpy as np
import nibabel as nib

from intrab.model.inferer import Inferer
from intrab.prompts.prompt import PromptStep, merge_prompt_steps
from intrab.prompts.prompt_3d import get_pos_clicks3D, get_bbox3d
from intrab.prompts.prompt_utils import (
    box_interpolation,
    box_propagation,
    get_bbox3d_sliced,
    get_fg_points_from_cc_centers,
    get_minimal_boxes_row_major,
    get_pos_clicks2D_row_major,
    get_seed_boxes,
    get_seed_point,
    interpolate_points,
    point_interpolation,
    point_propagation,
)

from intrab.utils.analysis import compute_dice


# ToDo: Save the Prompt before feeding into the model.
#   Also add a check to see if another model received the same Prompt.
#   If so, then we can just load the saved Prompt and compare with the same prompt.


class Prompter:
    is_static: bool = True

    def __init__(self, inferer: Inferer, seed: int = 11111):
        self.inferer: Inferer = inferer
        self.groundtruth: None = None
        self.seed = seed
        self.name = self.__class__.__name__

    def set_groundtruth(self, groundtruth: np.ndarray) -> None:
        """
        Sets the groundtruth that we want to predict.
        :param groundtruth: np.ndarray (Binary groundtruth mask)
        :return None
        """
        # Load the groundtruth
        self.groundtruth = groundtruth

    def predict_image(self, image_path: Path) -> tuple[nib.Nifti1Image, dict[int, np.ndarray]]:
        """Generate segmentation given prompt-style and model behavior."""
        # If the groundtruth is all zeros, return an empty mask
        if np.all(self.groundtruth == 0):
            img: nib.Nifti1Image = nib.load(image_path)
            binary_gt = np.zeros_like(img.get_fdata())
            empty_gt = nib.Nifti1Image(binary_gt.astype(np.uint8), img.affine)
            return empty_gt, None

        # Else predict the image
        self.inferer.set_image(image_path)
        prompt = self.get_prompt()
        return self.inferer.predict(prompt)

    @abstractmethod
    def get_prompt(self) -> PromptStep:
        pass


class NPointsPer2DSlicePrompter(Prompter):
    def __init__(self, inferer: Inferer, seed: int = 11111, n_points_per_slice: int = 5):
        super().__init__(inferer, seed)
        self.n_points_per_slice = n_points_per_slice

    def get_prompt(self) -> PromptStep:
        """
        Generate segmentation given prompt-style and model behavior.
        :return: str (Path to the predicted segmentation)
        """
        # Maybe name this SlicePrompts  to be less ambiguous
        return get_pos_clicks2D_row_major(self.groundtruth, self.n_points_per_slice, self.seed)


class PointInterpolationPrompter(Prompter):
    def __init__(self, inferer: Inferer, seed: int = 11111, n_slice_point_interpolation: int = 5):
        super().__init__(inferer, seed)
        self.n_slice_point_interpolation = n_slice_point_interpolation

    def get_prompt(self) -> PromptStep:
        """
        Generate segmentation given prompt-style and model behavior.
        :return: str (Path to the predicted segmentation)
        """
        return point_interpolation(gt=self.groundtruth, n_slices=self.n_slice_point_interpolation)


class PointPropagationPrompter(Prompter):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11111,
        n_seed_points_point_propagation: int = 5,
        n_points_propagation: int = 5,
    ):
        super().__init__(inferer, seed)
        self.n_seed_points_point_propagation = n_seed_points_point_propagation
        self.n_points_propagation = n_points_propagation

    def get_prompt(self) -> PromptStep:
        """
        Generate segmentation given prompt-style and model behavior.
        :return: str (Path to the predicted segmentation)
        """
        seed_points_prompt = get_seed_point(self.groundtruth, self.n_seed_points_point_propagation, self.seed)
        slices_to_infer = np.where(np.any(self.groundtruth, axis=(1, 2)))[0]

        all_point_prompts: PromptStep = point_propagation(
            self.inferer,
            seed_points_prompt,
            slices_to_infer,
            self.seed,
            self.n_points_propagation,
        )
        # use_point_prompt holds the points that were used in each slice, and originate from the seed prompt.
        return all_point_prompts


class BoxPer2DSlicePrompter(Prompter):

    def get_prompt(self) -> PromptStep:
        """
        Generate segmentation given prompt-style and model behavior.
        :return: str (Path to the predicted segmentation)
        """

        return get_minimal_boxes_row_major(self.groundtruth)


class BoxPer2dSliceFrom3DBoxPrompter(Prompter):

    def get_prompt(self) -> PromptStep:

        return get_bbox3d_sliced(self.groundtruth)


class BoxInterpolationPrompter(Prompter):

    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11111,
        n_slice_box_interpolation: int = 5,
    ):
        super().__init__(inferer, seed)
        self.n_slice_box_interpolation = n_slice_box_interpolation

    def get_prompt(self) -> PromptStep:
        box_seed_prompt: PromptStep = get_seed_boxes(self.groundtruth, self.n_slice_box_interpolation)
        return box_interpolation(box_seed_prompt)


class BoxPropagationPrompter(Prompter):

    def get_prompt(self) -> tuple[nib.Nifti1Image, dict[int, np.ndarray]]:

        median_box_seed_prompt: PromptStep = get_seed_boxes(self.groundtruth, 1)
        slices_to_infer = np.where(np.any(self.groundtruth, axis=(1, 2)))[0]
        return box_propagation(self.inferer, median_box_seed_prompt, slices_to_infer)


class NPoints3DVolumePrompter(Prompter):

    def __init__(self, inferer: Inferer, seed: int = 11111, n_points: int = 5):
        super().__init__(inferer, seed)
        self.n_points = n_points

    def get_prompt(self) -> tuple[nib.Nifti1Image, dict[int, np.ndarray]]:
        return get_pos_clicks3D(self.groundtruth, n_clicks=self.n_points, seed=self.seed)


class Box3DVolumePrompter(Prompter):

    def get_prompt(self) -> tuple[nib.Nifti1Image, dict[int, np.ndarray]]:
        return get_bbox3d(self.groundtruth)


static_prompt_styles = Literal[
    "NPointsPer2DSlicePrompter",
    "PointInterpolationPrompter",
    "PointPropagationPrompter",
    "BoxPer2DSlicePrompter",
    "BoxPer2dSliceFrom3DBoxPrompter",
    "BoxInterpolationPrompter",
    "BoxPropagationPrompter",
    "NPoints3DVolumePrompter",
    "Box3DVolumePrompter",
]