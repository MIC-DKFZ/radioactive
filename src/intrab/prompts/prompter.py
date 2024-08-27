from abc import abstractmethod
from pathlib import Path
from typing import Literal
import numpy as np

from intrab.model.inferer import Inferer
from intrab.prompts.prompt import PromptStep
from intrab.prompts.prompt_utils import (
    box_interpolation,
    box_propagation,
    get_bbox3d_sliced,
    get_fg_points_from_cc_centers,
    get_minimal_boxes_row_major,
    get_pos_clicks2D_row_major,
    get_seed_boxes,
    get_seed_point,
    point_interpolation,
    point_propagation,
)

from nibabel import Nifti1Image

# ToDo: Save the Prompt before feeding into the model.
#   Also add a check to see if another model received the same Prompt.
#   If so, then we can just load the saved Prompt and compare with the same prompt.


class Prompter:
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

    @abstractmethod
    def predict_image(self, image_path: Path) -> tuple[Nifti1Image, dict[int, np.ndarray]]:
        """Generate segmentation given prompt-style and model behavior."""
        pass


class NPointsPer2DSlicePrompter(Prompter):
    def __init__(self, inferer: Inferer, seed: int = 11111, n_points_per_slice: int = 5):
        super().__init__(inferer, seed)
        self.n_points_per_slice = n_points_per_slice

    def predict_image(self, image_path: Path) -> tuple[Nifti1Image, dict[int, np.ndarray]]:
        """
        Generate segmentation given prompt-style and model behavior.
        :return: str (Path to the predicted segmentation)
        """
        self.inferer.set_image(image_path)
        # Maybe name this SlicePrompts  to be less ambiguous
        point_prompts_step: PromptStep = get_pos_clicks2D_row_major(
            self.groundtruth, self.n_points_per_slice, self.seed
        )
        return self.inferer.predict(point_prompts_step)


class PointInterpolationPrompter(Prompter):
    def __init__(self, inferer: Inferer, seed: int = 11111, n_slice_point_interpolation: int = 5):
        super().__init__(inferer, seed)
        self.n_slice_point_interpolation = n_slice_point_interpolation

    def predict_image(self, image_path: Path) -> tuple[Nifti1Image, dict[int, np.ndarray]]:
        """
        Generate segmentation given prompt-style and model behavior.
        :return: str (Path to the predicted segmentation)
        """
        self.inferer.set_image(image_path)
        fg_points = get_fg_points_from_cc_centers(self.groundtruth, self.n_slice_point_interpolation, self.seed)
        point_prompts: PromptStep = point_interpolation(prompts=fg_points)

        return self.inferer.predict(point_prompts)


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

    def predict_image(self, image_path: Path) -> tuple[Nifti1Image, dict[int, np.ndarray]]:
        """
        Generate segmentation given prompt-style and model behavior.
        :return: str (Path to the predicted segmentation)
        """
        self.inferer.set_image(image_path)
        seed_points_prompt = get_seed_point(self.groundtruth, self.n_seed_points_point_propagation, self.seed)
        slices_to_infer = np.where(np.any(self.groundtruth, axis=(1, 2)))[0]

        all_point_prompts: PromptStep = point_propagation(
            self.inferer,
            seed_points_prompt,
            slices_to_infer,
            self.seed,
            self.n_points_propagation,
            verbose=False,
        )
        # use_point_prompt holds the points that were used in each slice, and originate from the seed prompt.
        return self.inferer.predict(all_point_prompts)


class BoxPer2DSlice(Prompter):

    def predict_image(self, image_path: Path) -> tuple[Nifti1Image, dict[int, np.ndarray]]:
        """
        Generate segmentation given prompt-style and model behavior.
        :return: str (Path to the predicted segmentation)
        """
        self.inferer.set_image(image_path)

        prompts = get_minimal_boxes_row_major(self.groundtruth)

        # use_point_prompt holds the points that were used in each slice, and originate from the seed prompt.
        return self.inferer.predict(prompts)


class BoxPer2dSliceFrom3DBox(Prompter):

    def predict_image(self, image_path: Path) -> tuple[Nifti1Image, dict[int, np.ndarray]]:
        self.inferer.set_image(image_path)

        prompts = get_bbox3d_sliced(self.groundtruth)

        # use_point_prompt holds the points that were used in each slice, and originate from the seed prompt.
        return self.inferer.predict(prompts)


class BoxInterpolationPrompter(Prompter):

    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11111,
        n_slice_box_interpolation: int = 5,
    ):
        super().__init__(inferer, seed)
        self.n_slice_box_interpolation = n_slice_box_interpolation

    def predict_image(self, image_path: Path) -> tuple[Nifti1Image, dict[int, np.ndarray]]:
        self.inferer.set_image(image_path)

        box_seed_prompt: PromptStep = get_seed_boxes(self.groundtruth, self.n_slice_box_interpolation)
        prompts = box_interpolation(box_seed_prompt)

        # use_point_prompt holds the points that were used in each slice, and originate from the seed prompt.
        return self.inferer.predict(prompts)


class BoxPropagation(Prompter):

    def predict_image(self, image_path: Path) -> tuple[Nifti1Image, dict[int, np.ndarray]]:
        self.inferer.set_image(image_path)

        median_box_seed_prompt: PromptStep = get_seed_boxes(self.groundtruth, 1)
        slices_to_infer = np.where(np.any(self.groundtruth, axis=(1, 2)))[0]
        all_box_prompts = box_propagation(self.inferer, median_box_seed_prompt, slices_to_infer)
        return self.inferer.predict(all_box_prompts)


static_prompt_styles = Literal[
    "NPointsPer2DSlicePrompter",
    "PointInterpolationPrompter",
    "PointPropagationPrompter",
    "BoxPer2DSlice",
    "BoxPer2dSliceFrom3DBox",
    "BoxInterpolationPrompter",
    "BoxPropagation",
]
