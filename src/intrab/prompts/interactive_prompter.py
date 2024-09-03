from abc import abstractmethod
from pathlib import Path
from typing import Literal
import numpy as np
from click import prompt
from intrab.model.inferer import Inferer
from intrab.prompts.prompt import PromptStep, merge_prompt_steps
from intrab.prompts.prompt_3d import get_pos_clicks3D, isolate_patch_around_point, obtain_misclassified_point_prompt
from intrab.prompts.prompt_utils import get_pos_clicks2D_row_major, get_fg_points_from_cc_centers, interpolate_points, get_middle_seed_point
from intrab.prompts.prompter import Prompter
from intrab.utils.interactivity import gen_contour_fp_scribble
from intrab.utils.result_data import PromptResult
import nibabel as nib
from loguru import logger
from rich.prompt import Prompt
from triton.language.extra.cuda import num_threads


class InteractivePrompter(Prompter):
    always_pass_prev_prompts = False
    is_static = False

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
    
    @abstractmethod
    def clear_states(self) -> None:
        """Clear any states saved in generating the image"""
        pass

    @abstractmethod
    def get_initial_prompt_step(self) -> PromptStep:
        """Gets the initial prompt for the image from the groundtruth."""
        pass

    # ToDo: Think about if one should actually use np.ndarrays as parameters of this function.
    @abstractmethod
    def get_next_prompt_step(
        self, pred: np.ndarray, low_res_logits: np.ndarray, all_prompts: list[PromptStep]
    ) -> PromptStep:
        """Gets the next prompt for the image from the groundtruth."""
        pass

    def predict_image(self, image_path: Path) -> list[PromptResult]:
        """Predicts the image for multiple steps until the stopping criteria is met."""

        self.inferer.set_image(image_path)
        dof: int = 0
        num_iter: int = 0
        perf: float = 0

        all_prompt_results: list[PromptResult] = []
        prompt_step: PromptStep = self.get_initial_prompt_step()
        pred, logits = self.inferer.predict(prompt_step)

        perf = self.get_performance(pred)
        all_prompt_results.append(
            PromptResult(
                predicted_image=pred, logits=logits, perf=perf, dof=dof, n_step=num_iter, prompt_step=prompt_step
            )
        )

        while not self.stopping_criteria_met(dof, perf, num_iter):
            prompt_step = self.get_next_prompt_step(pred, logits, [ap.prompt_step for ap in all_prompt_results])
            # Option to never forget previous prompts but feed all of them again in one huge joint prompt.
            if self.always_pass_prev_prompts:
                prompt_step = merge_prompt_steps(prompt_step, all_prompt_results[-1].prompt_step)

            pred, logits = self.inferer.predict(prompt_step, prev_seg = pred)
            dof += prompt_step.get_dof()
            perf = self.get_performance(pred)
            all_prompt_results.append(
                PromptResult(
                    predicted_image=pred, logits=logits, prompt_step=prompt_step, perf=perf, n_step=num_iter, dof=dof
                )
            )
            num_iter += 1

        # Clear any states generated in inferring this image
        self.clear_states()
        return all_prompt_results


class threeDInteractivePrompter(InteractivePrompter):
    def __init__(
        self,
        inferer: Inferer,
        n_points: int,
        seed: int = 11121,
        dof_bound: int | None = 60,
        perf_bound: float | None = 0.85,
        max_iter: int | None = 10,
        isolate_around_initial_point_size: int = None
    ):
        super().__init__(inferer, seed, dof_bound, perf_bound, max_iter)
        self.n_points = n_points
        self.gt_to_compare = None
        self.isolate_around_initial_point_size = isolate_around_initial_point_size

    def get_initial_prompt_step(self) -> PromptStep:
        return get_pos_clicks3D(self.groundtruth_model, n_clicks=self.n_points, seed=self.seed)
    
    def process_gt_to_compare(self, gt, initial_prompt_step, isolate_around_initial_point_size):
        return isolate_patch_around_point(gt, initial_prompt_step, isolate_around_initial_point_size)

    def get_next_prompt_step(self, pred: np.ndarray, low_res_logits: np.ndarray, all_prompts: list[PromptStep]) -> PromptStep:
        pred, _ = self.inferer.transform_to_model_coords(pred, is_seg = True) # Transform to the coordinate system in which inference will occur
        
        if self.gt_to_compare is None:
            self.gt_to_compare = self.process_gt_to_compare(self.groundtruth_model, all_prompts[0], self.isolate_around_initial_point_size)

        new_prompt = obtain_misclassified_point_prompt(pred, self.gt_to_compare, self.seed)
        
        return new_prompt

    def clear_states(self):
        self.isolated_gt = None
        return 

# ToDo: Check prompts generated are of decent 'quality'
class twoDInteractivePrompter(InteractivePrompter):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        dof_bound: int | None = 60,
        perf_bound: float | None = 0.85,
        max_iter: int | None = 10,
        contour_distance=2,
        disk_size_range=(0, 0),
        scribble_length=0.6,
    ):
        super().__init__(inferer, seed, dof_bound, perf_bound, max_iter)
        # Scribble generating parameters
        self.contour_distance = contour_distance
        self.disk_size_range = disk_size_range
        self.scribble_length = scribble_length

        self.bottom_seed_prompt = None # Store these to avoid repeated computation
        self.top_seed_prompt = None

    @staticmethod
    def get_generate_negative_prompts_flag(fn_mask: np.ndarray, fp_mask: np.ndarray) -> bool:
        # Old method: See if a lot of the foreground has been segmented. Also, this is fora  generate positive prompts flag, not a generate negaitve prompts flag
        # fn_count = np.sum(fn_mask)

        # fg_count = np.sum(self.groundtruth)

        # generate_positive_prompts_prob = (
        #     fn_count / fg_count
        # )  # Generate positive prompts when much of the forget_generate_negative_prompts_flageground isn't segmented
        # generate_positive_prompts_flag = np.random.binomial(1, generate_positive_prompts_prob)

        # New method: just find if there are more fn or fp
        fn_count = np.sum(fn_mask)
        fp_count = np.sum(fp_mask)

        generate_negative_prompts_prob = fp_count / (fn_count + fp_count)
        generate_negative_prompts_flag = np.random.binomial(1, generate_negative_prompts_prob)
        return bool(generate_negative_prompts_flag)

    def generate_positive_promptstep(
        self,
        fn_mask: np.ndarray,
        slices_inferred: set,
    ) -> PromptStep:

        if self.bottom_seed_prompt is None or self.top_seed_prompt is None:
            self.bottom_seed_prompt, _, self.top_seed_prompt = get_fg_points_from_cc_centers(
                self.groundtruth_model, 3
            )

        # Try to find fp coord in the middle 40% axially of the volume
        new_middle_seed_prompt = get_middle_seed_point(fn_mask, slices_inferred)

        # Interpolate linearly from botom_seed-prompt to top_seed_prompt through the new middle prompt to get new positive prompts
        new_seed_prompt = np.vstack([self.bottom_seed_prompt, new_middle_seed_prompt, self.top_seed_prompt])
        new_coords = interpolate_points(new_seed_prompt, kind="linear").astype(int)
        new_coords = new_coords[:, [2, 1, 0]]  # zyx -> xyz
        new_positive_promptstep = PromptStep(point_prompts=(new_coords, [1] * len(new_coords)))
        return new_positive_promptstep

    def _get_max_fp_sagittal_slice_idx(self, fp_mask):
        axis = 1  # Can extend to also check when fixing axis 2
        fp_sums = np.sum(fp_mask, axis=tuple({0, 1, 2} - {axis}))
        max_fp_idx = np.argmax(fp_sums)

        return max_fp_idx

    def _generate_scribble(
        self, slice_gt, slice_seg
    ):  # Mostly a wrapper for gen_countour_fp_scribble, but handles an edge case where it fails
        if not np.any(
            slice_gt
        ):  # There is no gt in the slice, but lots of fps. For now just draw a vertical line down the column with the most fps
            fp_per_column = np.sum(slice_seg, axis=0)
            max_fp_column = np.argmax(fp_per_column)
            scribble = np.zeros_like(slice_seg)
            scribble[:, max_fp_column] = 1
        else:
            scribble = gen_contour_fp_scribble(
                slice_gt,
                slice_seg,
                self.contour_distance,
                self.disk_size_range,
                self.scribble_length,
                seed=self.seed,
                verbose=False,
            )

        return scribble

    def generate_negative_promptstep(self, groundtruth, pred, fp_mask):
        # Obtain contour scribble on worst sagittal slice
        max_fp_idx = self._get_max_fp_sagittal_slice_idx(fp_mask)

        slice_gt = groundtruth[:, max_fp_idx]
        slice_seg = pred[:, max_fp_idx]

        # Try to generate a 'scribble' containing lots of false positives
        scribble = self._generate_scribble(slice_gt, slice_seg)

        if scribble is None:  # Scribble generation failed, get random negative click instead
            logger.warning(
                "Generating negative prompt failed - this should not happen. Will generate positive prompt instead."
            )
            return None

        else:  # Otherwise subset scribble to false positives to generate new prompt
            scribble_coords = np.where(scribble)
            scribble_coords = np.array(scribble_coords).T

            # Obtain false positive points and make new prompt
            coords_transpose = scribble_coords.T
            # fmt: off
            is_fp_mask = slice_seg[*coords_transpose].astype(bool)
            # fmt: on
            fp_coords = scribble_coords[is_fp_mask]

            ## Position fp_coords back into original 3d coordinate system
            missing_axis = np.repeat(max_fp_idx, len(fp_coords))
            fp_coords_3d = np.vstack([fp_coords[:, 0], missing_axis, fp_coords[:, 1]]).T
            fp_coords_3d = fp_coords_3d[:, [2, 1, 0]]  # zyx -> xyz
            new_negative_promptstep = PromptStep(point_prompts=(fp_coords_3d, [0] * len(fp_coords_3d)))

            return new_negative_promptstep

    def get_next_prompt_step(
        self, pred: nib.Nifti1Image, low_res_logits: np.ndarray, all_prompts: list[PromptStep]
    ) -> PromptStep:
        pred, _ = self.inferer.transform_to_model_coords(pred, is_seg = True) # Transform to the coordinate system in which inference will occur
        fn_mask = (pred == 0) & (self.groundtruth_model == 1)
        fp_mask = (pred == 1) & (self.groundtruth_model == 0)

        generate_negative_prompts_flag = self.get_generate_negative_prompts_flag(fn_mask, fp_mask)

        prompt_gen_failed = False
        if generate_negative_prompts_flag:
            prompt_step = self.generate_negative_promptstep(self.groundtruth_model, pred, fp_mask)
            if prompt_step is None:
                prompt_gen_failed = True

        if not generate_negative_prompts_flag or prompt_gen_failed:
            slices_inferred = self.prompt_step_all.get_slices_to_infer()  # Get which slices have been inferred on
            prompt_step = self.generate_positive_promptstep(fn_mask, slices_inferred)

        return prompt_step
    
    def clear_states(self) -> None:
        self.top_seed_prompt = None
        self.bottom_seed_prompt = None


class NPointsPer2DSliceInteractivePrompter(twoDInteractivePrompter):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        dof_bound: int | None = 60,
        perf_bound: float | None = 0.85,
        max_iter: int | None = 10,
        n_points_per_slice: int = 5,
    ):
        super().__init__(inferer, seed, dof_bound, perf_bound, max_iter)
        self.n_points_per_slice = n_points_per_slice

    def get_initial_prompt_step(self) -> PromptStep:
        initial_prompt_step = get_pos_clicks2D_row_major(self.groundtruth_model, self.n_points_per_slice, self.seed)
        self.prompt_step_all = initial_prompt_step

        return initial_prompt_step


interactive_prompt_styles = Literal["NPointsPer2DSliceInteractivePrompter",
                                    "threeDInteractivePrompter"]
