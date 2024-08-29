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

from nibabel import Nifti1Image

from intrab.utils.interactivity import gen_contour_fp_scribble
from intrab.utils.result_data import PromptResult

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

    def predict_image(self, image_path: Path) -> tuple[Nifti1Image, dict[int, np.ndarray]]:
        """Generate segmentation given prompt-style and model behavior."""
        # If the groundtruth is all zeros, return an empty mask
        if np.all(self.groundtruth == 0):
            img = nib.load(image_path)
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

    def get_prompt(self) -> tuple[Nifti1Image, dict[int, np.ndarray]]:

        median_box_seed_prompt: PromptStep = get_seed_boxes(self.groundtruth, 1)
        slices_to_infer = np.where(np.any(self.groundtruth, axis=(1, 2)))[0]
        return box_propagation(self.inferer, median_box_seed_prompt, slices_to_infer)


class NPoints3DVolumePrompter(Prompter):

    def __init__(self, inferer: Inferer, seed: int = 11111, n_points: int = 5):
        super().__init__(inferer, seed)
        self.n_points = n_points

    def get_prompt(self) -> tuple[Nifti1Image, dict[int, np.ndarray]]:
        return get_pos_clicks3D(self.groundtruth, n_clicks=self.n_points, seed=self.seed)


class Box3DVolumePrompter(Prompter):

    def get_prompt(self) -> tuple[Nifti1Image, dict[int, np.ndarray]]:
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


class InteractivePrompter(Prompter):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        dof_bound: int | None = 60,
        perf_bound: float | None = 0.85,
        max_iter: int | None = 10,
    ):
        super().__init__(inferer, seed)
        self.dof_bound: int | None = dof_bound
        self.perf_bound: float | None = perf_bound
        self.max_iter: int | None = max_iter
        assert any(
            [self.dof_bound, self.perf_bound, self.max_iter]
        ), "At least one stopping criteria needs to be provided: 'dof', 'perf' or 'num_iter'."

    def stopping_criteria_met(self, dof: int, perf: float, num_iter: int) -> bool:
        """Check if the stopping criteria is met."""
        dof_met = False
        perf_met = False
        num_it_met = False

        if self.dof_bound is not None:
            dof_met = dof >= self.dof_bound
        if self.perf_bound is not None:
            perf_met = perf >= self.perf_bound
        if self.max_iter is not None:
            num_it_met = num_iter >= self.max_iter

        return dof_met or perf_met or num_it_met

    def get_performance(self, pred: np.ndarray) -> float:
        """Get the DICE between prediciton and groundtruths."""
        tps = np.sum(pred * self.groundtruth)
        fps = np.sum(pred * (1 - self.groundtruth))
        fns = np.sum((1 - pred) * self.groundtruth)
        dice = 2 * tps / (2 * tps + fps + fns)
        return dice

    @abstractmethod
    def get_initial_prompt_step(self) -> PromptStep:
        """Gets the initial prompt for the image from the groundtruth."""
        pass

    # ToDo: Think about if one should actually use np.ndarrays as parameters of this function.
    @abstractmethod
    def get_next_prompt_step(self, pred: np.ndarray, low_res_logits: np.ndarray) -> PromptStep:
        """Gets the next prompt for the image from the groundtruth."""
        pass

    def predict_image(self, image_path: Path) -> list[PromptResult]:
        """Predicts the image for multiple steps until the stopping criteria is met."""

        self.inferer.set_image(image_path)
        dof: int = 0
        num_iter: int = 0
        perf: float = 0

        prompt_step: PromptStep = self.get_initial_prompt_step()
        pred, logits = self.inferer.predict(prompt_step)

        # ToDo: Update the DoF calculations to be more accurate.
        perf = self.get_performance(pred)

        while not self.stopping_criteria_met(dof, perf, num_iter):
            prompt_step = self.get_next_prompt_step(pred, logits)
            pred, logits = self.inferer.predict(prompt_step)
            dof += prompt_step.get_dof()
            perf = self.get_performance(pred)
            num_iter += 1


# ToDo: Make InteractivePrompter class, abstracting most of this away with static methods
# ToDo: Check prompts generated are of decent 'quality'
class NPointsPer2DSliceInteractivePrompter(Prompter):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11111,
        n_points_per_slice: int = 5,
        dof_bound=60,
        perf_bound=0.9,
        max_iter=10,
        scribble_length=0.2,
        contour_distance=2,
        disk_size_range=(0, 0),
    ):
        """
        :param inferer: Inferer object
        :param seed: int
        :param n_points_per_slice: Number of points to click per slice (and per interaction loop)
        :param dof_bound: Maximum degrees of freedom prompts allowed for interactions
        :param
        """
        super().__init__(inferer, seed)
        self.n_points_per_slice = n_points_per_slice  # Everything except this should be inherited
        self.dof_bound, self.perf_bound = dof_bound, perf_bound
        self.scribble_length, self.contour_distance, self.disk_size_range = (
            scribble_length,
            contour_distance,
            disk_size_range,
        )
        self.max_iter = max_iter

        self.bottom_seed_prompt, _, self.top_seed_prompt = None, None, None

    def get_generate_negative_prompts_flag(self, fn_mask: np.ndarray, fp_mask: np.ndarray) -> bool:
        # Old method: See if a lot of the foreground has been segmented. Also, this is fora  generate positive prompts flag, not a generate negaitve prompts flag
        # fn_count = np.sum(fn_mask)

        # fg_count = np.sum(self.groundtruth)

        # generate_positive_prompts_prob = (
        #     fn_count / fg_count
        # )  # Generate positive prompts when much of the foreground isn't segmented
        # generate_positive_prompts_flag = np.random.binomial(1, generate_positive_prompts_prob)

        # New method: just find if there are more fn or fp
        fn_count = np.sum(fn_mask)
        fp_count = np.sum(fp_mask)

        generate_negative_prompts_prob = fp_count / (fn_count + fp_count)
        generate_negative_prompts_flag = np.random.binomial(1, generate_negative_prompts_prob)
        return generate_negative_prompts_flag

    def gen_new_positive_prompts(
        self,
        bottom_seed_prompt: np.ndarray,
        top_seed_prompt: np.ndarray,
        fn_mask: np.ndarray,
        slices_inferred: set,
        dof: int,
    ) -> PromptStep:

        # Try to find fp coord in the middle 40% axially of the volume.
        lower, upper = np.percentile(slices_inferred, [30, 70])
        fp_coords = np.vstack(np.where(fn_mask)).T
        middle_mask = (lower < fp_coords[:, 0]) & (
            fp_coords[:, 0] < upper
        )  # Mask to determine which false negatives lie between the 30th to 70th percentile
        if np.sum(middle_mask) == 0:
            middle_mask = np.ones(
                len(fp_coords), bool
            )  # If there are no false negatives in the middle, draw from all coordinates (unlikely given that there must be many)
        fp_coords = fp_coords[middle_mask, :]
        new_middle_seed_prompt = fp_coords[np.random.choice(len(fp_coords), 1)]
        dof += 3

        # Interpolate linearly from botom_seed-prompt to top_seed_prompt through the new middle prompt to get new positive prompts
        new_seed_prompt = np.vstack([bottom_seed_prompt, new_middle_seed_prompt, top_seed_prompt])
        new_coords = interpolate_points(new_seed_prompt, kind="linear").astype(int)
        new_coords = new_coords[:, [2, 1, 0]]  # zyx -> xyz
        new_prompt_step = PromptStep(point_prompts=new_coords)
        return new_prompt_step

    def get_max_fp_sagittal_slice_idx(self, fp_mask):
        axis = 1  # Can extend to also check when fixing axis 2
        fp_sums = np.sum(fp_mask, axis=tuple({0, 1, 2} - {axis}))
        max_fp_idx = np.argmax(fp_sums)

        return max_fp_idx

    def generate_scribble(
        self, slice_gt, slice_seg, fp_mask
    ):  # Mostly a wrapper for gen_countour_fp_scribble, but handles an edge case where it could fail
        if not np.any(
            slice_gt
        ):  # There is no gt in the slice, but lots of fps. For now just draw a vertical line down the column with the most fps
            fp_mask = (
                slice_seg  # All segmented fg voxels are false positives, slice_seg only contains false positives
            )
            fp_per_column = np.sum(fp_mask, axis=0)
            max_fp_column = np.argmax(fp_per_column)
            scribble = np.zeros_like(slice_seg)
            scribble[:, max_fp_column] = 1
        else:
            scribble = gen_contour_fp_scribble(
                slice_gt,
                fp_mask,
                self.contour_distance,
                self.disk_size_range,
                self.scribble_length,
                seed=self.seed,
                verbose=False,
            )

        return scribble

    def predict_image(self, image_path: Path) -> list[tuple[Nifti1Image, int]]:
        # Initialise tracking flags
        has_generated_positive_prompts = False

        # Obtain initial prompt, segmentation, logits and dof
        prompt_step: PromptStep = get_pos_clicks2D_row_major(  # Duplicate calculation - extract prompt from prompter?
            self.groundtruth, self.n_points_per_slice, self.seed
        )
        slices_inferred = prompt_step.slices_to_infer
        base_prompter = NPointsPer2DSlicePrompter(self.inferer, self.seed, self.n_points_per_slice)
        pred, logits = base_prompter.predict_image(image_path)

        dof = len(prompt_step.slices_to_infer) * self.n_points_per_slice * 3

        # Iterate

        for num_iter in range(self.max_iter):
            # Obtain masks for new prompt calculations
            fn_mask = (pred == 0) & (self.groundtruth == 1)
            fp_mask = (pred == 1) & (self.groundtruth == 0)

            generate_negative_prompts_flag = self.get_generate_negative_prompts_flag(fn_mask, fp_mask)

            if generate_negative_prompts_flag:
                # Obtain contour scribble on worst sagittal slice
                max_fp_idx = self.get_max_fp_sagittal_slice_idx(fp_mask)

                max_fp_slice = self.groundtruth[:, max_fp_idx]
                slice_seg = pred[:, max_fp_idx]

                # Try to generate a 'scribble' containing lots of false positives

                scribble = self.generate_scribble(max_fp_slice, slice_seg)

                if scribble is None:  # Scribble generation failed, get random negative click instead
                    generate_negative_prompts_flag = False
                else:  # Otherwise subset scribble to false positives to generate new prompt
                    scribble_coords = np.where(scribble)
                    scribble_coords = np.array(scribble_coords).T

                    # Obtain false positive points and make new prompt
                    is_fp_mask = slice_seg[*scribble_coords.T].astype(bool)
                    fp_coords = scribble_coords[is_fp_mask]

                    ## Position fp_coords back into original 3d coordinate system
                    missing_axis = np.repeat(max_fp_idx, len(fp_coords))
                    fp_coords_3d = np.vstack([fp_coords[:, 0], missing_axis, fp_coords[:, 1]]).T
                    fp_coords_3d = fp_coords_3d[:, [2, 1, 0]]  # zyx -> xyz
                    improve_slices = np.unique(fp_coords_3d[:, 2])
                    dof += 3 * 4  # To dicuss: assume drawing a scribble is as difficult as drawing four points

                    if pass_prev_prompts:  # new prompt includes old prompts
                        ## Add to old prompt
                        coords = np.concatenate([prompt.coords, fp_coords_3d], axis=0)
                        labels = np.concatenate([prompt.labels, [0] * len(fp_coords_3d)])
                        prompt = PromptStep(point_prompts=(coords, labels))

                        ## Subset to prompts only on the slices with new prompts
                        fix_slice_mask = np.isin(prompt.coords[:, 2], improve_slices)
                        new_prompt = PromptStep(point_prompts=(coords[fix_slice_mask], labels[fix_slice_mask]))
                    else:
                        new_prompt = PromptStep(point_prompts=(fp_coords_3d, [0] * len(fp_coords_3d)))

            if not generate_negative_prompts_flag:
                # If running for the first time, generate universal bottom and top seed prompts for interpolation
                if not has_generated_positive_prompts:
                    dof += 6  # Increase for needing to choose bottom_seed_prompt and top_seed_prompt
                    bottom_seed_prompt, _, top_seed_prompt = get_fg_points_from_cc_centers(self.groundtruth, 3)
                    has_generated_positive_prompts = True

                new_positive_prompts, dof = self.gen_new_positive_prompts(
                    pred, bottom_seed_prompt, top_seed_prompt, fn_mask, slices_inferred, dof
                )
                if self.inferer.pass_prev_prompts:
                    prompt_step = merge_prompt_steps(new_positive_prompts, prompt_step)
                else:
                    prompt_step = new_positive_prompts
