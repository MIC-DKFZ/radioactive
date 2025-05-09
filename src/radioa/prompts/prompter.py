from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Literal
import numpy as np
import nibabel as nib
import torch

from radioa.model.inferer import Inferer
from radioa.prompts.prompt import PromptStep, merge_sparse_prompt_steps
from radioa.prompts.prompt_3d import get_linearly_spaced_coords, get_pos_clicks3D, get_bbox3d, subset_points_to_box
from radioa.prompts.prompt_utils import (
    box_interpolation,
    box_propagation,
    get_bbox3d_sliced,
    get_fg_point_from_cc_center,
    get_fg_points_from_cc_centers,
    get_minimal_boxes_row_major,
    get_pos_clicks2D_row_major,
    get_n_pos_neg_clicks2D_row_major,
    get_seed_boxes,
    get_seed_point,
    interpolate_points,
    point_interpolation,
    point_propagation,
)

from radioa.utils.result_data import PromptResult
from radioa.utils.transforms import (
    canonical_to_orig_sparse_coords,
    orig_to_SAR_dense,
    transform_prompt_to_model_coords,
)
from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib

# ToDo: Save the Prompt before feeding into the model.
#   Also add a check to see if another model received the same Prompt.
#   If so, then we can just load the saved Prompt and compare with the same prompt.


class Prompter:
    is_static: bool = True
    num_iterations: int = 20

    def __init__(self, inferer: Inferer, seed: int = 11111):
        self.inferer: Inferer = inferer
        self.groundtruth_nib: None | nib.Nifti1Image = None
        self.groundtruth_model: np.ndarray = None
        self.groundtruth_orig: np.ndarray | torch.Tensor = None
        self.groundtruth_SAR: np.ndarray = None
        self.seed = seed
        self.name = self.__class__.__name__

        self.orig_affine: tuple[float, float, float] = None
        self.orig_shape: tuple[int, int, int] = None

        # Overwrite this if the prompt that get_prompt supplies is in model coordinates, not orig
        # like point/box propagation
        self.promptstep_in_model_coord_system = False

        # Set seed
        if self.seed is not None:
            np.random.seed(seed)

    def get_performance(self, pred: np.ndarray | nib.Nifti1Image) -> float:
        """Get the DICE between prediciton and groundtruths."""
        if isinstance(pred, nib.Nifti1Image):
            pred = pred.get_fdata()

        if torch.cuda.is_available():
            pred = torch.from_numpy(pred).cuda().to(torch.int8)
        else:
            pred = pred.astype(np.int8)

        tps = (pred * self.groundtruth_orig).sum()
        fps = (pred * (1 - self.groundtruth_orig)).sum()
        fns = ((1 - pred) * self.groundtruth_orig).sum()
        dice = 2 * tps / (2 * tps + fps + fns)
        return float(dice)

    def get_slices_to_infer(self) -> list[int]:
        """Get the slices to infer from the groundtruth."""
        return np.where(np.any(self.groundtruth_SAR, axis=(1, 2)))[0]

    def set_groundtruth(self, groundtruth: nib.Nifti1Image) -> None:
        """
        Sets the groundtruth that we want to predict.
        :param groundtruth: np.ndarray (Binary groundtruth mask)
        :return None
        """
        # Load the groundtruth in model or original spacing
        self.groundtruth_nib = groundtruth
        self.groundtruth_orig = groundtruth.get_fdata()
        self.groundtruth_model = self.inferer.get_transformed_groundtruth(groundtruth)
        # SAR == z, y, x -- so we get the z dim in first axis.
        self.groundtruth_SAR = orig_to_SAR_dense(groundtruth)[0]
        if torch.cuda.is_available():
            self.groundtruth_orig = torch.from_numpy(self.groundtruth_orig).cuda().to(torch.int8)

        self.orig_affine = self.groundtruth_nib.affine
        self.orig_shape = self.groundtruth_nib.shape

    def predict_image(self, image_path: Path) -> PromptResult:
        """Generate segmentation given prompt-style and model behavior."""
        # If the groundtruth is all zeros, return an empty mask
        if np.all(self.groundtruth_model == 0) and torch.all(self.groundtruth_orig == 0):
            img: nib.Nifti1Image = load_any_to_nib(image_path)
            binary_gt = np.zeros_like(img.get_fdata())
            empty_gt = nib.Nifti1Image(binary_gt.astype(np.uint8), img.affine)
            return PromptResult(predicted_image=empty_gt, logits=None, prompt_step=None, perf=0, n_step=0, dof=0)

        # Else predict the image
        self.inferer.set_image(image_path)
        prompt: PromptStep = self.get_prompt()
        pred: nib.Nifti1Image
        logits: np.ndarray
        pred, logits, _ = self.inferer.predict(
            prompt, promptstep_in_model_coord_system=self.promptstep_in_model_coord_system
        )
        perf = self.get_performance(pred.get_fdata())
        prompt_res = PromptResult(predicted_image=pred, logits=None, prompt_step=prompt, perf=perf, n_step=0, dof=0)
        return prompt_res

    @abstractmethod
    def get_prompt(self) -> PromptStep:
        pass

    def transform_prompt_to_original_coords(self, prompt_orig: PromptStep) -> PromptStep:
        sparse_transform = partial(
            canonical_to_orig_sparse_coords, orig_affine=self.orig_affine, orig_shape=self.orig_shape
        )
        return transform_prompt_to_model_coords(prompt_orig, sparse_transform, transform_reverses_order=False)


class NFGPointsPer2DSlicePrompter(Prompter, ABC):
    n_point_per_slice: int

    def __init__(self, inferer: Inferer, seed: int = 11111):
        super().__init__(inferer, seed)
        self.n_points_per_slice = self.n_point_per_slice

    def get_prompt(self) -> PromptStep:
        """
        Prompt by creating n randomly chosen foregroudn points per slice
        """
        # Maybe name this SlicePrompts  to be less ambiguous
        prompt_RAS = get_pos_clicks2D_row_major(self.groundtruth_SAR, self.n_points_per_slice, self.seed)
        prompt_orig = self.transform_prompt_to_original_coords(prompt_RAS)
        return prompt_orig


class OneFGPointsPer2DSlicePrompter(NFGPointsPer2DSlicePrompter):
    n_point_per_slice = 1


class TwoFGPointsPer2DSlicePrompter(NFGPointsPer2DSlicePrompter):
    n_point_per_slice = 2


class ThreeFGPointsPer2DSlicePrompter(NFGPointsPer2DSlicePrompter):
    n_point_per_slice = 3


class FiveFGPointsPer2DSlicePrompter(NFGPointsPer2DSlicePrompter):
    n_point_per_slice = 5


class TenFGPointsPer2DSlicePrompter(NFGPointsPer2DSlicePrompter):
    n_point_per_slice = 10


class AlternatingNPointsPer2DSlicePrompter(Prompter, ABC):
    n_points_per_slice: int

    def get_prompt(self) -> PromptStep:
        """
        Prompt by creating n randomly chosen foregroudn points per slice
        """
        # Maybe name this SlicePrompts  to be less ambiguous
        prompt_RAS = get_n_pos_neg_clicks2D_row_major(self.groundtruth_SAR, self.n_points_per_slice, self.seed)
        prompt_orig = self.transform_prompt_to_original_coords(prompt_RAS)
        return prompt_orig


class Alternating2PointsPer2DSlicePrompter(AlternatingNPointsPer2DSlicePrompter):
    n_points_per_slice: int = 2


class Alternating3PointsPer2DSlicePrompter(AlternatingNPointsPer2DSlicePrompter):
    n_points_per_slice: int = 3


class Alternating5PointsPer2DSlicePrompter(AlternatingNPointsPer2DSlicePrompter):
    n_points_per_slice: int = 5


class Alternating10PointsPer2DSlicePrompter(AlternatingNPointsPer2DSlicePrompter):
    n_points_per_slice: int = 10


class CenterPointPrompter(Prompter):
    def get_prompt(self) -> PromptStep:
        """
        Prompt by taking the largest connected component of each slice, taking its centroid and correcting it to the nearest foreground pixel
        """
        # Get fg slices
        volume_fg = np.where(self.groundtruth_SAR == 1)  # Get foreground indices (formatted as triple of arrays)
        volume_fg = np.array(volume_fg).T  # Reformat to numpy array of shape n_fg_voxels x 3

        fg_slices = np.unique(volume_fg[:, 0])

        slice_prompts = []
        for slice_idx in fg_slices:
            gt_slice = self.groundtruth_SAR[slice_idx]
            slice_prompt = get_fg_point_from_cc_center(gt_slice)
            slice_prompts.append(
                np.array([slice_prompt[1], slice_prompt[0], slice_idx])
            )  # reverse order of slice_prompt since it's an axial slice from SAR, and add 3d context to put prompt into xyz

        coords_RAS = np.array(slice_prompts)
        prompt_RAS = PromptStep(point_prompts=(coords_RAS, np.ones(len(coords_RAS))))
        prompt_orig = self.transform_prompt_to_original_coords(prompt_RAS)
        return prompt_orig


class PointInterpolationPrompter(Prompter, ABC):
    n_slice_point_interpolation: int

    def get_prompt(self) -> PromptStep:
        """
        Simulates the user clicking in the connected component's center of mass `n_slice_point_interpolation` times.
        Slices are selected equidistantly between min and max slices with foreground (if not contiguous defaults to closest neighbors).
        Then the points are interpolated between the slices centers and prompted to the model.

        :return: The PromptStep from the interpolation of the points.
        """
        max_possible_clicks = min(self.n_slice_point_interpolation, len(self.get_slices_to_infer()))
        prompt_RAS = point_interpolation(gt=self.groundtruth_SAR, n_slices=max_possible_clicks)
        prompt_orig = self.transform_prompt_to_original_coords(prompt_RAS)
        return prompt_orig


class ThreePointInterpolationPrompter(PointInterpolationPrompter):
    n_slice_point_interpolation: int = 3


class FivePointInterpolationPrompter(PointInterpolationPrompter):
    n_slice_point_interpolation: int = 5


class TenPointInterpolationPrompter(PointInterpolationPrompter):
    n_slice_point_interpolation: int = 10


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

        self.promptstep_in_model_coord_system = True  # Overwrite default

    def get_prompt(self) -> PromptStep:
        """
        Generate segmentation given prompt-style and model behavior.
        :return: str (Path to the predicted segmentation)
        """
        seed_points_prompt_RAS = get_seed_point(self.groundtruth_SAR, self.n_seed_points_point_propagation, self.seed)
        slices_to_infer = np.where(np.any(self.groundtruth_SAR, axis=(1, 2)))[
            0
        ]  # Warning: won't work if set image involves resampling along z dimension
        seed_points_prompt_orig = self.transform_prompt_to_original_coords(seed_points_prompt_RAS)

        all_point_prompts: PromptStep = point_propagation(
            self.inferer,
            seed_points_prompt_orig,
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

        prompt_RAS = get_minimal_boxes_row_major(self.groundtruth_SAR)
        prompt_orig = self.transform_prompt_to_original_coords(prompt_RAS)
        return prompt_orig


class BoxPer2dSliceFrom3DBoxPrompter(Prompter):

    def get_prompt(self) -> PromptStep:
        prompt_RAS = get_bbox3d_sliced(self.groundtruth_SAR)
        prompt_orig = self.transform_prompt_to_original_coords(prompt_RAS)
        return prompt_orig


class BoxInterpolationPrompter(Prompter, ABC):
    n_slice_box_interpolation: int

    def get_prompt(self) -> PromptStep:
        max_possible_clicks = min(self.n_slice_box_interpolation, len(self.get_slices_to_infer()))
        seed_prompt_RAS = get_seed_boxes(self.groundtruth_SAR, max_possible_clicks)
        prompt_RAS = box_interpolation(seed_prompt_RAS)
        prompt_orig = self.transform_prompt_to_original_coords(prompt_RAS)
        return prompt_orig

class ThreeBoxInterpolationPrompter_nonperfect5(BoxInterpolationPrompter):
    n_slice_box_interpolation: int = 3
    pixel_max_shift: int = 5

    def get_prompt(self) -> PromptStep:
        max_possible_clicks = min(self.n_slice_box_interpolation, len(self.get_slices_to_infer()))
        seed_prompt_RAS = get_seed_boxes(self.groundtruth_SAR, max_possible_clicks)

        # Get image dimensions (assumes groundtruth_SAR has shape [depth, height, width])
        _, img_height, img_width = self.groundtruth_SAR.shape

        noisy_boxes = {}

        for slice_idx, (x_min, y_min, x_max, y_max) in seed_prompt_RAS.boxes.items():
            # Generate random noise in range [-5, 5]
            noise = np.random.randint(-self.pixel_max_shift, self.pixel_max_shift +1, size=4)

            # Apply noise
            x_min_noisy = x_min + noise[0]
            y_min_noisy = y_min + noise[1]
            x_max_noisy = x_max + noise[2]
            y_max_noisy = y_max + noise[3]

            # Ensure box remains within image boundaries
            x_min_noisy = np.clip(x_min_noisy, 0, img_width - 1)
            y_min_noisy = np.clip(y_min_noisy, 0, img_height - 1)
            x_max_noisy = np.clip(x_max_noisy, 0, img_width - 1)
            y_max_noisy = np.clip(y_max_noisy, 0, img_height - 1)

            # Ensure x_min < x_max and y_min < y_max
            if x_min_noisy >= x_max_noisy:
                x_min_noisy, x_max_noisy = max(0, x_min_noisy - 1), min(img_width - 1, x_max_noisy + 1)
            if y_min_noisy >= y_max_noisy:
                y_min_noisy, y_max_noisy = max(0, y_min_noisy - 1), min(img_height - 1, y_max_noisy + 1)

            noisy_boxes[slice_idx] = (x_min_noisy, y_min_noisy, x_max_noisy, y_max_noisy)

        # Create a new PromptStep with noisy boxes
        seed_prompt_RAS_noisy = PromptStep(box_prompts=noisy_boxes)

        prompt_RAS = box_interpolation(seed_prompt_RAS_noisy)
        prompt_orig = self.transform_prompt_to_original_coords(prompt_RAS)
        return prompt_orig

class ThreeBoxInterpolationPrompter_nonperfect3(ThreeBoxInterpolationPrompter_nonperfect5):
    n_slice_box_interpolation: int = 3
    pixel_max_shift: int = 3
class FiveBoxInterpolationPrompter(BoxInterpolationPrompter):
    n_slice_box_interpolation: int = 5

class ThreeBoxInterpolationPrompter(BoxInterpolationPrompter):
    n_slice_box_interpolation: int = 3

class TenBoxInterpolationPrompter(BoxInterpolationPrompter):
    n_slice_box_interpolation: int = 10


class BoxPropagationPrompter(Prompter):
    def __init__(self, inferer: Inferer, seed: int = 11111):
        super().__init__(inferer, seed)

        self.promptstep_in_model_coord_system = True  # Overwrite default

    def get_prompt(self) -> tuple[nib.Nifti1Image, dict[int, np.ndarray]]:
        median_box_seed_prompt_RAS: PromptStep = get_seed_boxes(self.groundtruth_SAR, 1)
        slices_to_infer = np.where(np.any(self.groundtruth_SAR, axis=(1, 2)))[0]

        median_box_seed_prompt_orig = self.transform_prompt_to_original_coords(median_box_seed_prompt_RAS)
        all_boxes_model = box_propagation(self.inferer, median_box_seed_prompt_orig, slices_to_infer)
        return all_boxes_model


class NPoints3DVolumePrompter(Prompter, ABC):
    n_points: int

    def __init__(self, inferer: Inferer, seed: int = 11111):
        super().__init__(inferer, seed)
        self.n_points = self.n_points

    def get_prompt(self) -> PromptStep:
        prompt_SAR = get_pos_clicks3D(self.groundtruth_SAR, n_clicks=self.n_points, seed=self.seed)
        # 3d functions don't reverse order of coords by themselves
        prompt_RAS = prompt_SAR
        prompt_RAS.coords = prompt_RAS.coords[:, ::-1]
        prompt_orig = self.transform_prompt_to_original_coords(prompt_RAS)
        return prompt_orig


class OnePoints3DVolumePrompter(NPoints3DVolumePrompter):
    n_points: int = 1


class TwoPoints3DVolumePrompter(NPoints3DVolumePrompter):
    n_points: int = 2


class ThreePoints3DVolumePrompter(NPoints3DVolumePrompter):
    n_points: int = 3


class FivePoints3DVolumePrompter(NPoints3DVolumePrompter):
    n_points: int = 5


class TenPoints3DVolumePrompter(NPoints3DVolumePrompter):
    n_points: int = 10


class CentroidPoint3DVolumePrompter(Prompter):
    def get_prompt(self) -> PromptStep:
        fg_coords = np.argwhere(self.groundtruth_SAR)
        centroid_SAR = fg_coords.mean(axis=0)
        centroid_RAS = centroid_SAR[::-1][None]
        centroid_prompt_RAS = PromptStep(point_prompts=(centroid_RAS, [1]))
        prompt_orig = self.transform_prompt_to_original_coords(centroid_prompt_RAS)
        return prompt_orig


class NPointsFromCenterCropped3DVolumePrompter(Prompter, ABC):
    n_points: int

    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11111,
        n_slice_point_interpolation: int = 5,
        isolate_around_initial_point_size: tuple[int, int, int] = None,
    ):
        super().__init__(inferer, seed)
        self.n_slice_point_interpolation = n_slice_point_interpolation
        self.isolate_around_initial_point_size = isolate_around_initial_point_size
        if self.isolate_around_initial_point_size is not None:
            self.isolate_around_initial_point_size = np.array(isolate_around_initial_point_size)
        self.promptstep_in_model_coord_system = True

    def get_prompt(self) -> PromptStep:

        max_possible_clicks = min(self.n_slice_point_interpolation, len(self.get_slices_to_infer()))
        prompt_RAS = point_interpolation(gt=self.groundtruth_SAR, n_slices=max_possible_clicks)
        prompt_orig = self.transform_prompt_to_original_coords(prompt_RAS)
        prompt_model = self.inferer.transform_promptstep_to_model_coords(prompt_orig)

        # Must subset so that everything lies in a patch. Take a crop around the centroid of the prompts
        if self.isolate_around_initial_point_size is not None:
            prompt_model = subset_points_to_box(prompt_model, self.isolate_around_initial_point_size)

        # Now sample regularly
        prompt_sub = get_linearly_spaced_coords(prompt_model, self.n_points)

        return prompt_sub


class OnePointsFromCenterCropped3DVolumePrompter(NPointsFromCenterCropped3DVolumePrompter):
    n_points = 1


class TwoPointsFromCenterCropped3DVolumePrompter(NPointsFromCenterCropped3DVolumePrompter):
    n_points = 2


class ThreePointsFromCenterCropped3DVolumePrompter(NPointsFromCenterCropped3DVolumePrompter):
    n_points = 3


class FivePointsFromCenterCropped3DVolumePrompter(NPointsFromCenterCropped3DVolumePrompter):
    n_points = 5


class TenPointsFromCenterCropped3DVolumePrompter(NPointsFromCenterCropped3DVolumePrompter):
    n_points = 10


class Box3DVolumePrompter(Prompter):

    def get_prompt(self) -> tuple[nib.Nifti1Image, dict[int, np.ndarray]]:
        prompt_RAS = get_bbox3d(self.groundtruth_SAR)
        prompt_orig = self.transform_prompt_to_original_coords(prompt_RAS)
        return prompt_orig


static_prompt_styles = Literal[
    # ------------------------- 2D Positive Click Prompters ------------------------- #
    "OneFGPointsPer2DSlicePrompter",
    "TwoFGPointsPer2DSlicePrompter",
    "ThreeFGPointsPer2DSlicePrompter",
    "FiveFGPointsPer2DSlicePrompter",
    "TenFGPointsPer2DSlicePrompter",
    "CenterPointPrompter",
    "Alternating2PointsPer2DSlicePrompter",
    "Alternating3PointsPer2DSlicePrompter",
    "Alternating5PointsPer2DSlicePrompter",
    "Alternating10PointsPer2DSlicePrompter",
    # -------------------- 2D Point Interpolation and propagation ------------------- #
    "ThreePointInterpolationPrompter",
    "FivePointInterpolationPrompter",
    "TenPointInterpolationPrompter",
    "PointPropagationPrompter",
    'ThreeBoxInterpolationPrompter_nonperfect3',
    'ThreeBoxInterpolationPrompter_nonperfect5',
    # ------------------------------- 2D Box prompters ------------------------------ #
    "BoxPer2DSlicePrompter",
    "BoxPer2dSliceFrom3DBoxPrompter",
    # --------------------- 2D Box Interpolation and Propagation -------------------- #
    "ThreeBoxInterpolationPrompter",
    "FiveBoxInterpolationPrompter",
    "TenBoxInterpolationPrompter",
    "BoxPropagationPrompter",
    # ------------------------- 3D Volume Prompters ------------------------- #
    "OnePoints3DVolumePrompter",
    "TwoPoints3DVolumePrompter",
    "ThreePoints3DVolumePrompter",
    "FivePoints3DVolumePrompter",
    "TenPoints3DVolumePrompter",
    "CentroidPoint3DVolumePrompter",
    "OnePointsFromCenterCropped3DVolumePrompter",
    "TwoPointsFromCenterCropped3DVolumePrompter",
    "ThreePointsFromCenterCropped3DVolumePrompter",
    "FivePointsFromCenterCropped3DVolumePrompter",
    "TenPointsFromCenterCropped3DVolumePrompter",
    "Box3DVolumePrompter",
]
