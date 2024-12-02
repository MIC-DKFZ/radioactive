from abc import abstractmethod
from pathlib import Path
from typing import Literal
import numpy as np
from click import prompt
from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib
from radioa.model.SAMMed3D import SAMMed3DInferer
from radioa.model.inferer import Inferer
from radioa.prompts.prompt import PromptStep, merge_sparse_prompt_steps
from radioa.prompts.prompt_3d import (
    get_linearly_spaced_coords,
    get_pos_clicks3D,
    isolate_patch_around_point,
    obtain_misclassified_point_prompt_2d,
    obtain_misclassified_point_prompt_3d,
    subset_points_to_box,
)
from radioa.prompts.prompt_utils import (
    box_interpolation,
    get_fg_point_from_cc_center,
    get_n_largest_CCs,
    get_pos_clicks2D_row_major,
    get_fg_points_from_cc_centers,
    get_seed_boxes,
    get_seed_point,
    interpolate_points,
    get_middle_seed_point,
    point_interpolation,
    point_propagation,
)
from radioa.prompts.prompter import Prompter
from radioa.utils.image import get_crop_pad_params_from_gt_or_prompt
from radioa.utils.interactivity import gen_contour_fp_scribble
from radioa.utils.result_data import PromptResult
import nibabel as nib
from loguru import logger
from rich.prompt import Prompt
from triton.language.extra.cuda import num_threads
from scipy.ndimage import center_of_mass


class InteractivePrompter(Prompter):
    always_pass_prev_prompts = False
    is_static = False

    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        dof_bound: int | None = None,
        perf_bound: float | None = None,
        max_iter: int | None = None,
    ):
        super().__init__(inferer, seed)

        self.promptstep_in_model_coord_system = (
            True  # Expect promptsteps to generally be supplied in model coord system for interactive prompters.
        )

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
        # If the groundtruth is all zeros, return an empty mask
        if np.all(self.groundtruth_model == 0):
            img: nib.Nifti1Image = load_any_to_nib(image_path)
            binary_gt = np.zeros_like(img.get_fdata())
            empty_gt = nib.Nifti1Image(binary_gt.astype(np.uint8), img.affine)
            return [
                PromptResult(predicted_image=empty_gt, logits=None, prompt_step=None, perf=0, n_step=0, dof=0)
            ] * (
                self.max_iter + 1
            )  # return list of expected length

        # Else predict the image
        self.inferer.set_image(image_path)
        dof: int = 0
        num_iter: int = 0
        perf: float = 0

        all_prompt_results: list[PromptResult] = []
        prompt_step: PromptStep = self.get_initial_prompt_step()
        pred, logits, _ = self.inferer.predict(
            prompt_step, promptstep_in_model_coord_system=self.promptstep_in_model_coord_system
        )

        perf = self.get_performance(pred)
        all_prompt_results.append(
            PromptResult(
                predicted_image=pred, logits=None, perf=perf, dof=dof, n_step=num_iter, prompt_step=prompt_step
            )
        )

        while not self.stopping_criteria_met(dof, perf, num_iter):
            prompt_step = self.get_next_prompt_step(pred, logits, [ap.prompt_step for ap in all_prompt_results])
            if prompt_step is None:
                break  # None is returned if prediction is perfect
            # Option to never forget previous prompts but feed all of them again in one huge joint prompt.
            if self.always_pass_prev_prompts:
                merged_prompt_step = merge_sparse_prompt_steps(
                    [prompt_step, all_prompt_results[-1].prompt_step]
                )  # Merge sparse prompts
                merged_prompt_step.set_masks(prompt_step.masks)  # Take dense prompts from current promptstep
                prompt_step = merged_prompt_step

            pred, logits, _ = self.inferer.predict(
                prompt_step, prev_seg=pred, promptstep_in_model_coord_system=self.promptstep_in_model_coord_system
            )
            dof += prompt_step.get_dof()
            perf = self.get_performance(pred)
            all_prompt_results.append(
                PromptResult(
                    predicted_image=pred, logits=None, prompt_step=prompt_step, perf=perf, n_step=num_iter, dof=dof
                )
            )
            num_iter += 1

            if perf == 1:
                break  # Crazy, but happened with sam2

        if len(all_prompt_results) < self.max_iter + 1:
            all_prompt_results.extend(
                [all_prompt_results[-1]] * (self.max_iter + 1 - len(all_prompt_results))
            )  # Repeat last result so we have self.max_iter+1 results
        return all_prompt_results


# ToDo: Check prompts generated are of decent 'quality'
class twoDInteractivePrompter(InteractivePrompter):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        n_ccs_positive_interaction: int = 1,
        dof_bound: int | None = None,
        perf_bound: float | None = None,
        max_iter: int | None = None,
        contour_distance=2,
        disk_size_range=(0, 0),
        scribble_length=0.6,
    ):
        super().__init__(inferer, seed, dof_bound, perf_bound, max_iter)
        # Positive scribble generating parameters
        self.n_ccs_positive_interaction = n_ccs_positive_interaction
        # Negative scribble generating parameters
        self.contour_distance = contour_distance
        self.disk_size_range = disk_size_range
        self.scribble_length = scribble_length

        self.negative_failed_choose_n_points = 3  # Default

    @staticmethod
    def get_generate_negative_prompts_flag(fn_mask: np.ndarray, fp_mask: np.ndarray) -> bool:
        # Find if there are more fn or fp
        fn_count = np.sum(fn_mask)
        fp_count = np.sum(fp_mask)

        if fn_count == 0:
            return True
        elif fp_count == 0:
            return False

        generate_negative_prompts_prob = fp_count / (fn_count + fp_count)
        generate_negative_prompts_flag = np.random.binomial(1, generate_negative_prompts_prob)
        return bool(generate_negative_prompts_flag)

    def generate_positive_promptstep(self, fn_mask: np.ndarray, n_ccs: int = 1) -> PromptStep:

        largest_CCs = get_n_largest_CCs(fn_mask, n_ccs)

        centroids = []
        for cc in largest_CCs:  # Loop through largest CCs, adding components as you go
            slices_to_consider = np.where(np.any(cc, axis=(1, 2)))[0]

            for z in slices_to_consider:
                CC_slice = cc[z]
                centroid = get_fg_point_from_cc_center(CC_slice)
                centroid = [z, centroid[0], centroid[1]]
                centroids.append(centroid)

        centroids = np.array(centroids)

        # No need to subset to voxels not yet segmented since these are drawn from false negatives
        # centroids = centroids[:, [2, 1, 0]]  # zyx -> xyz
        new_positive_promptstep = PromptStep(point_prompts=(centroids, [1] * len(centroids)))
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

        if (
            scribble is None
        ):  # if this fails go to exception mode - just choose self.negative_failed_choose_n points instead

            return get_pos_clicks3D(fp_mask, self.negative_failed_choose_n_points, self.seed)

        else:  # Otherwise subset scribble to false positives to generate new prompt
            scribble_coords = np.where(scribble)
            scribble_coords = np.array(scribble_coords).T

            # Obtain false positive points and make new prompt
            coords_transpose = scribble_coords.T
            # fmt: off
            is_fp_mask = slice_seg[tuple(coords_transpose)].astype(bool)
            # fmt: on
            fp_coords = scribble_coords[is_fp_mask]

            if (
                len(fp_coords) == 0
            ):  # if no fp_coords generated go to exception mode - just choose self.negative_failed_choose_n points instead
                return get_pos_clicks3D(fp_mask, self.negative_failed_choose_n_points, self.seed)

            ## Position fp_coords back into original 3d coordinate system
            missing_axis = np.repeat(max_fp_idx, len(fp_coords))
            fp_coords_3d = np.vstack([fp_coords[:, 0], missing_axis, fp_coords[:, 1]]).T
            # fp_coords_3d = fp_coords_3d[:, [2, 1, 0]]  # zyx -> xyz
            new_negative_promptstep = PromptStep(point_prompts=(fp_coords_3d, [0] * len(fp_coords_3d)))

            return new_negative_promptstep

    def get_next_prompt_step(
        self, pred: nib.Nifti1Image, low_res_logits: np.ndarray, all_prompts: list[PromptStep]
    ) -> PromptStep:
        pred, _ = self.inferer.transform_to_model_coords_dense(
            pred, is_seg=True
        )  # Transform to the coordinate system in which inference will occur
        if np.all(pred == self.groundtruth_model):
            return None
        fn_mask = (pred == 0) & (self.groundtruth_model == 1)
        fp_mask = (pred == 1) & (self.groundtruth_model == 0)

        generate_negative_prompts_flag = self.get_generate_negative_prompts_flag(fn_mask, fp_mask)

        prompt_gen_failed = False
        if generate_negative_prompts_flag:
            prompt_step = self.generate_negative_promptstep(self.groundtruth_model, pred, fp_mask)
            if prompt_step is None:
                # logger.debug('generate negative prompts failed')
                prompt_gen_failed = True

        if not generate_negative_prompts_flag or prompt_gen_failed:  # Get which slices have been inferred on
            prompt_step = self.generate_positive_promptstep(fn_mask, self.n_ccs_positive_interaction)

        # ####
        # # Debug:
        # aligned_coords = prompt_step.coords[:, [2,1,0]] # back to zyx
        # true_labels = self.groundtruth_model[tuple(aligned_coords.T)]
        # prompt_type = 'positive' if not generate_negative_prompts_flag or prompt_gen_failed else 'negative'
        # logger.debug(f'{prompt_type} prompts created. # of corrective prompts: {len(true_labels)}, # of mistakes {np.sum(true_labels!=prompt_step.labels)}')
        # ####

        prompt_step.set_masks(low_res_logits)

        return prompt_step


class threeDCroppedInteractivePrompterNoPrevPoint(InteractivePrompter):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        n_points: int = 1,
        dof_bound: int | None = None,
        perf_bound: float | None = None,
        max_iter: int | None = None,
        isolate_around_initial_point_size: int = None,
    ):
        super().__init__(inferer, seed, dof_bound, perf_bound, max_iter)
        self.n_points = n_points
        self.isolate_around_initial_point_size = isolate_around_initial_point_size

    def get_initial_prompt_step(self) -> PromptStep:
        prompt_RAS = get_pos_clicks3D(self.groundtruth_SAR, n_clicks=self.n_points, seed=self.seed)
        prompt_RAS.coords = prompt_RAS.coords[
            :, ::-1
        ]  # This isn't done in the function since it's generally to be avoided for 3d models
        prompt_orig = self.transform_prompt_to_original_coords(prompt_RAS)
        prompt_model = self.inferer.transform_promptstep_to_model_coords(prompt_orig)
        return prompt_model

    def process_seg_to_compare(
        self,
        seg: np.ndarray,
        initial_prompt_step: PromptStep,
        isolate_around_initial_point_size: tuple[int, int, int],
    ):
        return isolate_patch_around_point(seg, initial_prompt_step, isolate_around_initial_point_size).astype(int)

    def get_next_prompt_step(
        self, pred_model: np.ndarray, low_res_logits: np.ndarray, all_prompts: list[PromptStep]
    ) -> PromptStep:
        gt_to_compare = (
            self.process_seg_to_compare(  # obtain cropped (or, rather, isolated) gt (repeated calculation)
                self.groundtruth_model, all_prompts[0], self.isolate_around_initial_point_size
            )
        )

        pred_to_compare = self.process_seg_to_compare(
            pred_model, all_prompts[0], self.isolate_around_initial_point_size
        )  # Both the pred to compare and the gt to compare must match outside the isolated region

        new_prompt = obtain_misclassified_point_prompt_3d(pred_to_compare, gt_to_compare, self.seed)
        new_prompt.set_masks(low_res_logits)

        return new_prompt

    def predict_image(
        self, image_path: Path
    ) -> list[PromptResult]:  # Same as in InteractivePrompter, except need to keep track of crop_pad_params.
        """Predicts the image for multiple steps until the stopping criteria is met."""
        # If the groundtruth is all zeros, return an empty mask
        if np.all(self.groundtruth_model == 0):
            img: nib.Nifti1Image = load_any_to_nib(image_path)
            binary_gt = np.zeros_like(img.get_fdata())
            empty_gt = nib.Nifti1Image(binary_gt.astype(np.uint8), img.affine)
            return [
                PromptResult(predicted_image=empty_gt, logits=None, prompt_step=None, perf=0, n_step=0, dof=0)
            ] * (
                self.max_iter + 1
            )  # return list of expected length

        # Else predict the image
        self.inferer.set_image(image_path)
        dof: int = 0
        num_iter: int = 0
        perf: float = 0

        all_prompt_results: list[PromptResult] = []
        prompt_step: PromptStep = self.get_initial_prompt_step()
        crop_pad_params = get_crop_pad_params_from_gt_or_prompt(self.inferer.img, prompt_step)
        pred, logits, pred_model = self.inferer.predict(
            prompt_step, crop_pad_params, promptstep_in_model_coord_system=self.promptstep_in_model_coord_system
        )

        perf = self.get_performance(pred)
        all_prompt_results.append(
            PromptResult(
                predicted_image=pred, logits=None, perf=perf, dof=dof, n_step=num_iter, prompt_step=prompt_step
            )
        )

        while not self.stopping_criteria_met(dof, perf, num_iter):
            all_prompt_steps = [ap.prompt_step for ap in all_prompt_results]
            prompt_step = self.get_next_prompt_step(pred_model, logits, all_prompt_steps)
            # Option to never forget previous prompts but feed all of them again in one huge joint prompt.
            if self.always_pass_prev_prompts:
                prompt_step = merge_sparse_prompt_steps([prompt_step, all_prompt_results[-1].prompt_step])

            pred, logits, pred_model = self.inferer.predict(
                prompt_step, crop_pad_params, promptstep_in_model_coord_system=self.promptstep_in_model_coord_system
            )
            dof += prompt_step.get_dof()
            perf = self.get_performance(pred)
            all_prompt_results.append(
                PromptResult(
                    predicted_image=pred, logits=None, prompt_step=prompt_step, perf=perf, n_step=num_iter, dof=dof
                )
            )
            num_iter += 1

        return all_prompt_results


class threeDCroppedFromCenterInteractivePrompterNoPrevPoint(threeDCroppedInteractivePrompterNoPrevPoint):
    """
    Same as threeDCroppedInteractivePrompterNoPrevPoint except that instead of the initial point being a random fg point, it's a center point as determined by the point interpolation method.
    """

    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        n_points: int = 1,
        dof_bound: int | None = None,
        perf_bound: float | None = None,
        max_iter: int | None = None,
        isolate_around_initial_point_size: int = None,
        n_slice_point_interpolation: int = None,
    ):
        super().__init__(
            inferer=inferer,
            seed=seed,
            n_points=n_points,
            dof_bound=dof_bound,
            perf_bound=perf_bound,
            max_iter=max_iter,
            isolate_around_initial_point_size=isolate_around_initial_point_size,
        )
        self.n_slice_point_interpolation = n_slice_point_interpolation

    def get_initial_prompt_step(self) -> PromptStep:
        max_possible_clicks = min(self.n_slice_point_interpolation, len(self.get_slices_to_infer()))
        prompt_RAS = point_interpolation(gt=self.groundtruth_SAR, n_slices=max_possible_clicks)
        prompt_orig = self.transform_prompt_to_original_coords(prompt_RAS)
        prompt_model = self.inferer.transform_promptstep_to_model_coords(prompt_orig)

        # Must subset so that everything lies in a patch. Take a crop around the centroid of the prompts
        prompt_cropped = subset_points_to_box(prompt_model, self.isolate_around_initial_point_size)

        # Now sample regularly
        prompt_sub = get_linearly_spaced_coords(prompt_cropped, self.n_points)

        return prompt_sub


class threeDCroppedFromCenterAnd2dAlgoInteractivePrompterNoPrevPoint(
    threeDCroppedFromCenterInteractivePrompterNoPrevPoint
):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        n_points: int = 1,
        dof_bound: int | None = None,
        perf_bound: float | None = None,
        max_iter: int | None = None,
        isolate_around_initial_point_size: int = None,
        n_slice_point_interpolation: int = 5,
        num_corrective_prompts: int = 1,
        n_ccs_positive_interaction: int = 1,
        contour_distance=2,
        disk_size_range=(0, 0),
        scribble_length=0.6,
    ):
        super().__init__(
            inferer,
            seed,
            n_points=n_points,
            dof_bound=dof_bound,
            perf_bound=perf_bound,
            max_iter=max_iter,
            isolate_around_initial_point_size=isolate_around_initial_point_size,
            n_slice_point_interpolation=n_slice_point_interpolation,
        )
        self.num_corrective_prompts = num_corrective_prompts
        self.n_ccs_positive_interaction = n_ccs_positive_interaction

        self.contour_distance = contour_distance
        self.disk_size_range = disk_size_range
        self.scribble_length = scribble_length
        self.negative_failed_choose_n_points = 5

    @staticmethod
    def get_generate_negative_prompts_flag(fn_mask: np.ndarray, fp_mask: np.ndarray) -> bool:
        # Find if there are more fn or fp
        fn_count = np.sum(fn_mask)
        fp_count = np.sum(fp_mask)

        generate_negative_prompts_prob = fp_count / (fn_count + fp_count)
        generate_negative_prompts_flag = np.random.binomial(1, generate_negative_prompts_prob)
        return bool(generate_negative_prompts_flag)

    def generate_positive_promptstep(self, fn_mask: np.ndarray, n_ccs: int = 1) -> PromptStep:

        largest_CCs = get_n_largest_CCs(fn_mask, n_ccs)

        centroids = []
        for cc in largest_CCs:  # Loop through largest CCs, adding components as you go
            slices_to_consider = np.where(np.any(cc, axis=(1, 2)))[0]

            for z in slices_to_consider:
                CC_slice = cc[z]
                centroid = get_fg_point_from_cc_center(CC_slice)
                centroid = [z, centroid[0], centroid[1]]
                centroids.append(centroid)

        centroids = np.array(centroids)

        # No need to subset to voxels not yet segmented since these are drawn from false negatives

        # Align back to RAS format and return
        # centroids = centroids[:, [2, 1, 0]]  # zyx -> xyz # Don't do this in 3d- we don't inherit from 2d natural vision.
        new_positive_promptstep = PromptStep(point_prompts=(centroids, [1] * len(centroids)))
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
            return get_pos_clicks3D(fp_mask, self.negative_failed_choose_n_points, self.seed)

        else:
            scribble_coords = np.where(
                scribble
            )  # Otherwise subset scribble to false positives to generate new prompt
            scribble_coords = np.array(scribble_coords).T

            # Obtain false positive points and make new prompt
            coords_transpose = scribble_coords.T
            # fmt: off
            is_fp_mask = slice_seg[tuple(coords_transpose)].astype(bool)

            # fmt: on
            fp_coords = scribble_coords[is_fp_mask]

            if (
                len(fp_coords) == 0
            ):  # if no fp_coords generated go to exception mode - just choose self.negative_failed_choose_n points instead
                return get_pos_clicks3D(fp_mask, self.negative_failed_choose_n_points, self.seed)

            ## Position fp_coords back into original 3d coordinate system
            missing_axis = np.repeat(max_fp_idx, len(fp_coords))
            fp_coords_3d = np.vstack([fp_coords[:, 0], missing_axis, fp_coords[:, 1]]).T
            # fp_coords_3d = fp_coords_3d[:, [2, 1, 0]]  # zyx -> xyz # Don't do this in 3D - we don't inherit fron 2d natural vision
            new_negative_promptstep = PromptStep(point_prompts=(fp_coords_3d, [0] * len(fp_coords_3d)))

            return new_negative_promptstep

    def get_next_prompt_step(self, pred_model: np.ndarray, low_res_logits: np.ndarray, all_prompts: list[PromptStep]):
        # Ensure that gt and pred can only differ inside a self.isolate_around_initial_point_size crop centered
        # around the initial prompt

        gt_to_compare = (
            self.process_seg_to_compare(  # obtain cropped (or, rather, isolated) gt (repeated calculation)
                self.groundtruth_model, all_prompts[0], self.isolate_around_initial_point_size
            )
        )

        pred_to_compare = self.process_seg_to_compare(
            pred_model, all_prompts[0], self.isolate_around_initial_point_size
        )

        ## Now use 2d algorithm logic to generate the negative/positive prompts
        fn_mask = (pred_to_compare == 0) & (gt_to_compare == 1)
        fp_mask = (pred_to_compare == 1) & (gt_to_compare == 0)

        generate_negative_prompts_flag = self.get_generate_negative_prompts_flag(fn_mask, fp_mask)

        prompt_gen_failed = False
        if generate_negative_prompts_flag:
            prompt_step = self.generate_negative_promptstep(gt_to_compare, pred_to_compare, fp_mask)
            if prompt_step is None:
                # logger.debug('generate negative prompts failed')
                prompt_gen_failed = True

        if not generate_negative_prompts_flag or prompt_gen_failed:  # Get which slices have been inferred on
            prompt_step = self.generate_positive_promptstep(fn_mask, self.n_ccs_positive_interaction)

        # 3d models don't need as many points as 2d; subset to the desired number.
        coords, labels = prompt_step.coords, prompt_step.labels
        num_corrective_prompts = min(self.num_corrective_prompts, len(coords))
        sample_inds = np.random.choice(range(len(coords)), num_corrective_prompts, replace=False)
        coords = coords[sample_inds]
        labels = labels[sample_inds]
        prompt_step = PromptStep(point_prompts=(coords, labels))

        # ####
        # # Debug:
        # aligned_coords = prompt_step.coords[:, [2,1,0]] # back to zyx
        # true_labels = self.groundtruth_model[tuple(aligned_coords.T)]
        # prompt_type = 'positive' if not generate_negative_prompts_flag or prompt_gen_failed else 'negative'
        # logger.debug(f'{prompt_type} prompts created. # of corrective prompts: {len(true_labels)}, # of mistakes {np.sum(true_labels!=prompt_step.labels)}')
        # ####

        prompt_step.set_masks(low_res_logits)

        return prompt_step


class twoD1PointUnrealisticInteractivePrompterNoPrevPoint(InteractivePrompter):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        dof_bound: int | None = None,
        perf_bound: float | None = None,
        max_iter: int | None = None,
        n_init_points_per_slice: int = 1,
    ):

        super().__init__(inferer, seed, dof_bound, perf_bound, max_iter)
        self.n_init_points_per_slice = n_init_points_per_slice

    def get_initial_prompt_step(self) -> PromptStep:
        prompt_RAS = get_pos_clicks2D_row_major(self.groundtruth_SAR, self.n_init_points_per_slice, self.seed)
        prompt_orig = self.transform_prompt_to_original_coords(prompt_RAS)
        prompt_model = self.inferer.transform_promptstep_to_model_coords(prompt_orig)
        return prompt_model

    def get_next_prompt_step(
        self, pred: np.ndarray, low_res_logits: np.ndarray, all_prompts: list[PromptStep]
    ) -> PromptStep:
        pred, _ = self.inferer.transform_to_model_coords_dense(pred, is_seg=True)
        slices_inferred = all_prompts[0].get_slices_to_infer()
        all_slice_prompt_steps = []

        for slice_idx in slices_inferred:
            slice_seg = pred[slice_idx]
            slice_gt = self.groundtruth_model[slice_idx]
            if not np.all(slice_seg == slice_gt):  # check for perfect prediction. Crazy, but happened with
                slice_prompt_step = obtain_misclassified_point_prompt_2d(slice_seg, slice_gt, slice_idx, self.seed)
                all_slice_prompt_steps.append(slice_prompt_step)

        new_prompt_step = merge_sparse_prompt_steps(all_slice_prompt_steps)
        new_prompt_step.coords = new_prompt_step.coords[:, ::-1]  # Desperate fix attempt
        new_prompt_step.set_masks(low_res_logits)

        return new_prompt_step


class OnePointPer2DSliceInteractivePrompterNoPrevPoint(twoDInteractivePrompter):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        n_ccs_positive_interaction: int = 1,
        dof_bound: int | None = None,
        perf_bound: float | None = None,
        max_iter: int | None = None,
        n_init_points_per_slice: int = 1,
    ):
        super().__init__(inferer, seed, n_ccs_positive_interaction, dof_bound, perf_bound, max_iter)
        self.n_init_points_per_slice = n_init_points_per_slice

    def get_initial_prompt_step(self) -> PromptStep:
        prompt_RAS = get_pos_clicks2D_row_major(self.groundtruth_SAR, self.n_init_points_per_slice, self.seed)
        prompt_orig = self.transform_prompt_to_original_coords(prompt_RAS)
        prompt_model = self.inferer.transform_promptstep_to_model_coords(prompt_orig)

        return prompt_model


class PointInterpolationInteractivePrompterNoPrevPoint(twoDInteractivePrompter):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        n_ccs_positive_interaction: int = 1,
        dof_bound: int | None = None,
        perf_bound: float | None = None,
        max_iter: int | None = None,
        n_slice_point_interpolation: int = 5,
    ):
        super().__init__(inferer, seed, n_ccs_positive_interaction, dof_bound, perf_bound, max_iter)
        self.n_slice_point_interpolation = n_slice_point_interpolation

    def get_initial_prompt_step(self) -> PromptStep:
        """
        Simulates the user clicking in the connected component's center of mass `n_slice_point_interpolation` times.
        Slices are selected equidistantly between min and max slices with foreground (if not contiguous defaults to closest neighbors).
        Then the points are interpolated between the slices centers and prompted to the model.

        :return: The PromptStep from the interpolation of the points.
        """
        max_possible_clicks = min(self.n_slice_point_interpolation, len(self.get_slices_to_infer()))
        prompt_RAS = point_interpolation(gt=self.groundtruth_SAR, n_slices=max_possible_clicks)
        prompt_orig = self.transform_prompt_to_original_coords(prompt_RAS)
        prompt_model = self.inferer.transform_promptstep_to_model_coords(prompt_orig)
        return prompt_model


class BoxInterpolationInteractivePrompterNoPrevPoint(twoDInteractivePrompter):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        n_ccs_positive_interaction: int = 1,
        dof_bound: int | None = None,
        perf_bound: float | None = None,
        max_iter: int | None = None,
    ):
        super().__init__(inferer, seed, n_ccs_positive_interaction, dof_bound, perf_bound, max_iter)
        self.n_slice_box_interpolation = 3

    def get_initial_prompt_step(self) -> PromptStep:
        """
        Simulates the user clicking in the connected component's center of mass `n_slice_point_interpolation` times.
        Slices are selected equidistantly between min and max slices with foreground (if not contiguous defaults to closest neighbors).
        Then the points are interpolated between the slices centers and prompted to the model.

        :return: The PromptStep from the interpolation of the points.
        """
        max_possible_clicks = min(self.n_slice_box_interpolation, len(self.get_slices_to_infer()))
        seed_prompt_RAS = get_seed_boxes(self.groundtruth_SAR, max_possible_clicks)
        prompt_RAS = box_interpolation(seed_prompt_RAS)
        prompt_orig = self.transform_prompt_to_original_coords(prompt_RAS)
        prompt_model = self.inferer.transform_promptstep_to_model_coords(prompt_orig)
        return prompt_model


class PointPropagationInteractivePrompterNoPrevPoint(twoDInteractivePrompter):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        n_ccs_positive_interaction: int = 1,
        dof_bound: int | None = None,
        perf_bound: float | None = None,
        max_iter: int | None = None,
        n_seed_points_point_propagation: int = 5,
        n_points_propagation: int = 5,
    ):
        super().__init__(inferer, seed, n_ccs_positive_interaction, dof_bound, perf_bound, max_iter)
        self.n_points_propagation = n_points_propagation
        self.n_seed_points_point_propagation = n_seed_points_point_propagation

    def get_initial_prompt_step(self) -> PromptStep:
        """
        Simulates the user clicking in the connected component's center of mass `n_slice_point_interpolation` times.
        Slices are selected equidistantly between min and max slices with foreground (if not contiguous defaults to closest neighbors).
        Then the points are interpolated between the slices centers and prompted to the model.

        :return: The PromptStep from the interpolation of the points.
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
        return all_point_prompts


# ----------------------------------------
# Same prompters, but passing previous points
class threeDCroppedInteractivePrompterWithPrevPoint(threeDCroppedInteractivePrompterNoPrevPoint):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        n_points: int = 1,
        dof_bound: int | None = None,
        perf_bound: float | None = None,
        max_iter: int | None = None,
        isolate_around_initial_point_size: int = None,
    ):
        super().__init__(inferer, seed, n_points, dof_bound, perf_bound, max_iter, isolate_around_initial_point_size)
        self.always_pass_prev_prompts = True


class threeDCroppedFromCenterInteractivePrompterWithPrevPoint(threeDCroppedFromCenterInteractivePrompterNoPrevPoint):
    """
    Same as threeDCroppedInteractivePrompterNoPrevPoint except that instead of the initial point being a random fg point, it's a center point as determined by the point interpolation method.
    """

    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        n_points: int = 1,
        dof_bound: int | None = None,
        perf_bound: float | None = None,
        max_iter: int | None = None,
        isolate_around_initial_point_size: int = None,
        n_slice_point_interpolation: int = 5,
    ):
        super().__init__(
            inferer=inferer,
            seed=seed,
            n_points=n_points,
            dof_bound=dof_bound,
            perf_bound=perf_bound,
            max_iter=max_iter,
            isolate_around_initial_point_size=isolate_around_initial_point_size,
            n_slice_point_interpolation=n_slice_point_interpolation,
        )
        self.always_pass_prev_prompts = True


class threeDCroppedFromCenterAnd2dAlgoInteractivePrompterWithPrevPoint(
    threeDCroppedFromCenterAnd2dAlgoInteractivePrompterNoPrevPoint
):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        n_points: int = 1,
        dof_bound: int | None = None,
        perf_bound: float | None = None,
        max_iter: int | None = None,
        isolate_around_initial_point_size: int = None,
        n_slice_point_interpolation: int = 5,
        num_corrective_prompts: int = 1,
        n_ccs_positive_interaction: int = 1,
    ):
        super().__init__(
            inferer,
            seed,
            n_points,
            dof_bound,
            perf_bound,
            max_iter,
            isolate_around_initial_point_size,
            n_slice_point_interpolation,
            num_corrective_prompts,
            n_ccs_positive_interaction,
        )

        self.always_pass_prev_prompts = True


class twoD1PointUnrealisticInteractivePrompterWithPrevPoint(twoD1PointUnrealisticInteractivePrompterNoPrevPoint):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        dof_bound: int | None = None,
        perf_bound: float | None = None,
        max_iter: int | None = None,
        n_init_points_per_slice: int = 1,
    ):

        super().__init__(inferer, seed, dof_bound, perf_bound, max_iter, n_init_points_per_slice)
        self.always_pass_prev_prompts = True


class OnePointPer2DSliceInteractivePrompterWithPrevPoint(OnePointPer2DSliceInteractivePrompterNoPrevPoint):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        n_ccs_positive_interaction: int = 1,
        dof_bound: int | None = None,
        perf_bound: float | None = None,
        max_iter: int | None = None,
        n_init_points_per_slice: int = 1,
    ):
        super().__init__(
            inferer, seed, n_ccs_positive_interaction, dof_bound, perf_bound, max_iter, n_init_points_per_slice
        )
        self.always_pass_prev_prompts = True


class PointInterpolationInteractivePrompterWithPrevPoint(PointInterpolationInteractivePrompterNoPrevPoint):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        n_ccs_positive_interaction: int = 1,
        dof_bound: int | None = None,
        perf_bound: float | None = None,
        max_iter: int | None = None,
        n_slice_point_interpolation: int = 5,
    ):
        super().__init__(
            inferer, seed, n_ccs_positive_interaction, dof_bound, perf_bound, max_iter, n_slice_point_interpolation
        )
        self.always_pass_prev_prompts = True


class PointPropagationInteractivePrompterWithPrevPoint(PointPropagationInteractivePrompterNoPrevPoint):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        n_ccs_positive_interaction: int = 1,
        dof_bound: int | None = None,
        perf_bound: float | None = None,
        max_iter: int | None = None,
        n_seed_points_point_propagation: int = 5,
        n_points_propagation: int = 5,
    ):
        super().__init__(
            inferer,
            seed,
            n_ccs_positive_interaction,
            dof_bound,
            perf_bound,
            max_iter,
            n_seed_points_point_propagation,
            n_points_propagation,
        )
        self.always_pass_prev_prompts = True


interactive_prompt_styles = Literal[
    "OnePointPer2DSliceInteractivePrompterNoPrevPoint",
    "OnePointPer2DSliceInteractivePrompterWithPrevPrompt",
    "PointInterpolationInteractivePrompterNoPrevPoint",
    "PointInterpolationInteractivePrompterWithPrevPoint",
    "BoxInterpolationInteractivePrompterNoPrevPoint",
    "PointPropagationInteractivePrompterNoPrevPoint",
    "PointPropagationInteractivePrompterWithPrevPoint",
    "threeDCroppedInteractivePrompterNoPrevPoint",
    "threeDCroppedInteractivePrompterWithPrevPoint",
    "threeDCroppedFromCenterInteractivePrompterNoPrevPoint",
    "threeDCroppedFromCenterInteractivePrompterWithPrevPoint",
    "threeDCroppedFromCenterAnd2dAlgoInteractivePrompterNoPrevPoint",
    "threeDCroppedFromCenterAnd2dAlgoInteractivePrompterWithPrevPoint",
    "twoD1PointUnrealisticInteractivePrompterNoPrevPoint",
    "twoD1PointUnrealisticInteractivePrompterWithPrevPoint",
]
