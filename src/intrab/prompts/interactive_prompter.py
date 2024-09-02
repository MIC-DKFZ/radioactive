from abc import abstractmethod
from pathlib import Path
from typing import Literal
import numpy as np
from intrab.model.inferer import Inferer
from intrab.prompts.prompt import PromptStep, merge_prompt_steps
from intrab.prompts.prompt_utils import get_fg_points_from_cc_centers, get_pos_clicks2D_row_major, interpolate_points
from intrab.prompts.prompter import Prompter
from intrab.utils.interactivity import gen_contour_fp_scribble
from intrab.utils.result_data import PromptResult
from nibabel import Nifti1Image


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

    def predict_image(self, image_path: Path) -> list[tuple[nib.Nifti1Image, int]]:
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


# ToDo: Check prompts generated are of decent 'quality'
class InteractivePrompterMIC(InteractivePrompter):
    def __init__(
        self,
        inferer: Inferer,
        seed: int = 11121,
        dof_bound: int | None = 60,
        perf_bound: float | None = 0.85,
        max_iter: int | None = 10,
    ):
        super().__init__(inferer, seed, dof_bound, perf_bound, max_iter)
        self.initial_dof = None

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

    def generate_positive_promptstep(self, fn_mask: np.ndarray, slices_inferred: set, dof: int) -> PromptStep:

        if not self.has_generated_positive_prompts:
            self.bottom_seed_prompt, _, self.top_seed_prompt = get_fg_points_from_cc_centers(self.groundtruth, 3)

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
        new_seed_prompt = np.vstack([self.bottom_seed_prompt, new_middle_seed_prompt, self.top_seed_prompt])
        new_coords = interpolate_points(new_seed_prompt, kind="linear").astype(int)
        new_coords = new_coords[:, [2, 1, 0]]  # zyx -> xyz
        new_positive_promptstep = PromptStep(point_prompts=new_coords)
        return new_positive_promptstep

    def _get_max_fp_sagittal_slice_idx(self, fp_mask):
        axis = 1  # Can extend to also check when fixing axis 2
        fp_sums = np.sum(fp_mask, axis=tuple({0, 1, 2} - {axis}))
        max_fp_idx = np.argmax(fp_sums)

        return max_fp_idx

    def _generate_scribble(
        self, slice_gt, slice_seg, fp_mask
    ):  # Mostly a wrapper for gen_countour_fp_scribble, but handles an edge case where it fails
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

    def generate_negative_promptstep(self, groundtruth, pred, fp_mask):
        # Obtain contour scribble on worst sagittal slice
        max_fp_idx = self._get_max_fp_sagittal_slice_idx(fp_mask)

        max_fp_slice = groundtruth[:, max_fp_idx]
        slice_seg = pred[:, max_fp_idx]

        # Try to generate a 'scribble' containing lots of false positives
        scribble = self._generate_scribble(max_fp_slice, slice_seg)

        if scribble is None:  # Scribble generation failed, get random negative click instead
            return None, None

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
            new_negative_promptstep = PromptStep(points_prompts=(fp_coords_3d, [0] * len(fp_coords_3d)))

            return new_negative_promptstep

    def get_next_prompt_step(self, pred, dof):
        fn_mask = (pred == 0) & (self.groundtruth == 1)
        fp_mask = (pred == 1) & (self.groundtruth == 0)

        generate_negative_prompts_flag = self.get_generate_negative_prompts_flag(fn_mask, fp_mask)

        if generate_negative_prompts_flag:
            prompt_step = self.generate_negative_promptstep(self.groundtruth, pred, fp_mask)

            if prompt_step is None:
                prompt_gen_failed = True
            else:
                dof += 12  # ToDo discuss how many dofs to add here
                return prompt_step, dof

        if not generate_negative_prompts_flag or prompt_gen_failed:
            prompt_step = self.generate_positive_promptstep(pred, fn_mask)

            if not self.has_generated_positive_prompts:
                dof += 6  # When running for the first time, two additional points must be annotated
                self.has_generated_positive_prompts = True

            dof += 3

            return prompt_step, dof

    def predict_image(self, image_path: Path) -> tuple[list[Nifti1Image], list[int]]:
        # Initialise helper variables
        self.has_generated_positive_prompts = False
        num_iter = 0
        dof = 0
        results = []
        prompt_step_all = None

        # Obtain initial segmetnation
        self.inferer.set_image(image_path)

        prompt_step, initial_dof = self.get_initial_prompt_step()
        if self.inferer.pass_prev_prompts:
            prompt_step_all = prompt_step
        pred, logits = self.inferer.predict(prompt_step)

        dof = initial_dof
        perf = self.get_performance(pred)

        result = PromptResult(pred, logits, perf, num_iter, dof)
        results.append(result)

        # Iterate on initial segmentation
        while not self.stopping_criteria_met(dof, perf, num_iter):
            # Obtain next prompt_step
            prompt_step, dof = self.get_next_prompt_step(pred, logits)

            # Merge prompt step with previous steps if desired
            slices_with_new_prompts = prompt_step.get_slices_to_infer()

            if self.inferer.pass_prev_prompts:
                prompt_step_all = merge_prompt_steps(prompt_step, prompt_step_all)
                # coords_to_use, labels_to_use = prompt_step_all.coords[fix_slice_mask], prompt_step_all.labels[fix_slice_mask]
                # Check this new way works as well as using the fix_slice_mask
                coords_to_use, labels_to_use = (
                    prompt_step_all.coords[slices_with_new_prompts],
                    prompt_step_all.labels[slices_with_new_prompts],
                )
                prompt_step = PromptStep(point_prompts=(coords_to_use, labels_to_use))

            # Generate new segmentation and integrate into old one
            new_seg, new_logits = self.inferer.predict(prompt_step, logits, transform=False)
            pred[slices_with_new_prompts] = new_seg[slices_with_new_prompts]
            logits[slices_with_new_prompts] = new_logits[slices_with_new_prompts]

            perf = self.get_performance(pred)

            result = PromptResult(pred, logits, perf, num_iter, dof)
            results.append(result)

            num_iter += 1

        return results


class NPointsPer2DSliceInteractive(InteractivePrompterMIC):
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
        initial_prompt_step = get_pos_clicks2D_row_major(self.groundtruth, self.n_points_per_slice, self.seed)
        initial_dof = len(initial_prompt_step.get_slices_to_infer()) * 3  # 3 dofs per slice inferred

        return initial_prompt_step, initial_dof


interactive_prompt_styles = Literal["NPointsPer2DSliceInteractive",]


from abc import abstractmethod
from pathlib import Path
from typing import Literal
import numpy as np
from intrab.model.inferer import Inferer
from intrab.prompts.prompt import PromptStep, merge_prompt_steps
from intrab.prompts.prompt_utils import get_fg_points_from_cc_centers, get_pos_clicks2D_row_major, interpolate_points
from intrab.prompts.prompter import Prompter
from intrab.utils.interactivity import gen_contour_fp_scribble
from intrab.utils.result_data import PromptResult
from loguru import logger
from nibabel import Nifti1Image

class InteractivePrompter(Prompter):
    is_static:bool = False
    
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

        return (dof_met and perf_met) or num_it_met

    def get_performance(self, pred: np.ndarray) -> float:
        """Get the DICE between prediciton and groundtruths."""
        tps = np.sum(pred * self.groundtruth)
        fps = np.sum(pred * (1 - self.groundtruth))
        fns = np.sum((1 - pred) * self.groundtruth)
        dice = 2 * tps / (2 * tps + fps + fns)
        return dice

    @abstractmethod
    def get_initial_prompt_step(self) -> tuple[PromptStep, int]:
        """Gets the initial prompt for the image from the groundtruth."""
        pass

    # ToDo: Think about if one should actually use np.ndarrays as parameters of this function.
    @abstractmethod
    def get_next_prompt_step(self, pred: np.ndarray, low_res_logits: np.ndarray) -> PromptStep:
        """Gets the next prompt for the image from the groundtruth."""
        pass

    def predict_image(self, image_path: Path) -> tuple[list[Nifti1Image],list[PromptResult]]:
        """Predicts the image for multiple steps until the stopping criteria is met."""

        self.inferer.set_image(image_path)
        dof: int = 0
        num_iter: int = 0
        perf: float = 0

        prompt_step, dof = self.get_initial_prompt_step()
        pred, logits = self.inferer.predict(prompt_step)

        # ToDo: Update the DoF calculations to be more accurate.
        perf = self.get_performance(pred)

        while not self.stopping_criteria_met(dof, perf, num_iter):
            prompt_step = self.get_next_prompt_step(pred, logits)
            pred, logits = self.inferer.predict(prompt_step)
            dof += prompt_step.get_dof() # Won't work so easily right now; a scribble prompt is stored as a lot of points prompts, but the dof aren't the same as manually annotating a bunch of points
            perf = self.get_performance(pred)
            num_iter += 1


# ToDo: Check prompts generated are of decent 'quality'
class twoDInteractivePrompter(InteractivePrompter):
    def __init__(
            self,
            inferer: Inferer,
            seed: int = 11121,
            dof_bound: int | None = 60,
            perf_bound: float | None = 0.85,
            max_iter: int | None = 10,
            contour_distance = 2,
            disk_size_range = (0,0),
            scribble_length = 0.6

        ):
        super().__init__(inferer, seed, dof_bound, perf_bound, max_iter)
        self.initial_dof = None
        self.prompts = []
        self.prompt_step_all = None

        # Scribble generating parameters
        self.contour_distance = contour_distance
        self.disk_size_range = disk_size_range
        self.scribble_length = scribble_length
        
    
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
    
    def generate_positive_promptstep(
        self,
        fn_mask: np.ndarray,
        slices_inferred: set,
    ) -> PromptStep:
        
        if not self.has_generated_positive_prompts:
            self.bottom_seed_prompt, _, self.top_seed_prompt = get_fg_points_from_cc_centers(self.groundtruth, 3)

        # Try to find fp coord in the middle 40% axially of the volume.
        lower, upper = np.percentile(slices_inferred, [30, 70])
        fp_coords = np.vstack(np.where(fn_mask)).T
        middle_mask = (lower < fp_coords[:, 0]) & (
            fp_coords[:, 0] < upper
        )  # Mask to determine which false negatives lie between the 30th to 70th percentile
        if np.sum(middle_mask) == 0:
            logger.info('Failed to generate prompt in middle 40 percent of the volume. This may be worth checking out.')
            middle_mask = np.ones(
                len(fp_coords), bool
            )  # If there are no false negatives in the middle, draw from all coordinates (unlikely given that there must be many)

        fp_coords = fp_coords[middle_mask, :]
        new_middle_seed_prompt = fp_coords[np.random.choice(len(fp_coords), 1)]

        # Interpolate linearly from botom_seed-prompt to top_seed_prompt through the new middle prompt to get new positive prompts
        new_seed_prompt = np.vstack([self.bottom_seed_prompt, new_middle_seed_prompt, self.top_seed_prompt])
        new_coords = interpolate_points(new_seed_prompt, kind="linear").astype(int)
        new_coords = new_coords[:, [2, 1, 0]]  # zyx -> xyz
        new_positive_promptstep = PromptStep(point_prompts=(new_coords, [1]*len(new_coords)))
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

        if scribble is None: # Scribble generation failed, get random negative click instead
            logger.warning('Generating negative prompt failed - this should not happen. Will generate positive prompt instead.')
            return None

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
            new_negative_promptstep = PromptStep(point_prompts = (fp_coords_3d, [0] * len(fp_coords_3d)))

            return new_negative_promptstep

    def get_next_prompt_step(self, pred, dof, pass_prev_prompts):
        fn_mask = (pred == 0) & (self.groundtruth == 1)
        fp_mask = (pred == 1) & (self.groundtruth == 0)

        generate_negative_prompts_flag = self.get_generate_negative_prompts_flag(fn_mask, fp_mask)

        if generate_negative_prompts_flag:
            prompt_step = self.generate_negative_promptstep(self.groundtruth, pred, fp_mask)

            if prompt_step is None: 
                prompt_gen_failed = True
            else: 
                prompt_gen_failed = False
                # TESTING:
                if len(prompt_step.get_slices_to_infer()) < 5:
                    logger.warning('Negative prompt step is changing less than 5 slices; probably pretty ineffective')
                dof+=12 # ToDo discuss how many dofs to add here

        if not generate_negative_prompts_flag or prompt_gen_failed:
            slices_inferred = self.prompt_step_all.get_slices_to_infer() # Get which slices have been inferred on
            prompt_step = self.generate_positive_promptstep(
                fn_mask,
                slices_inferred
            )

            if not self.has_generated_positive_prompts:
                dof += 6 # When running for the first time, two additional points must be annotated
                self.has_generated_positive_prompts = True

            dof += 3

        # Handle prompt step storage and merging; as needed
        self.store_prompt_steps(prompt_step)

        if pass_prev_prompts:
            return self.prompt_step_all, dof
        else:
            return prompt_step, dof 
        
    def store_prompt_steps(self, prompt_step:PromptStep):
        # Store the prompt step as a state
        self.prompts.append(prompt_step) 

        # Merge into prompt_step_all
        if self.prompt_step_all is None:
            self.prompt_step_all = prompt_step
        else:
            self.prompt_step_all = merge_prompt_steps(self.prompt_step_all, prompt_step)

    def predict_and_merge_new_seg(self, pred: np.ndarray, logits:np.ndarray, prompt_step:PromptStep):
        """
        Wrapper for inferer.predict that handles merging new segmentations. 
        """
        
        # Obtain new prediction
        new_seg, new_logits = self.inferer.predict(prompt_step, logits, transform=False)

        # Merge into previous segmentation
        slices_with_new_prompts = prompt_step.get_slices_to_infer()

        pred[slices_with_new_prompts] = new_seg[slices_with_new_prompts]
        logits.update(new_logits)

        return pred, logits
    
    def process_results(self, pred, logits, perf, num_iter, dof) -> tuple[Nifti1Image, PromptMetaResult]:
        pred_orig_system = self.inferer.inv_trans(pred)
        meta_res = PromptMetaResult(logits, perf, num_iter, dof)

        return pred_orig_system, meta_res

    def predict_image(self, image_path: Path) -> tuple[list[Nifti1Image], list[PromptMetaResult]]:
        # Initialise helper variables
        self.has_generated_positive_prompts = False
        num_iter = 0
        dof = None
        preds = []
        preds_meta = []

        # Obtain initial segmetnation
        self.inferer.set_image(image_path)

        prompt_step, dof = self.get_initial_prompt_step()
        pred, logits = self.inferer.predict(prompt_step, transform = False)

        perf = self.get_performance(pred)

        pred_orig_system, meta_res = self.process_results(pred, logits, perf, num_iter, dof)
        preds.append(pred_orig_system)
        preds_meta.append(meta_res)

        # Iterate on initial segmentation
        while not self.stopping_criteria_met(dof, perf, num_iter):
            # Obtain next prompt_step 
            prompt_step, dof = self.get_next_prompt_step(pred, dof, self.inferer.pass_prev_prompts)

            # Generate new segmentation and integrate into old one
            pred, logits = self.predict_and_merge_new_seg(pred, logits, prompt_step)
            
            perf = self.get_performance(pred)

            pred_orig_system, meta_res = self.process_results(pred, logits, perf, num_iter, dof)
            preds.append(pred_orig_system)
            preds_meta.append(meta_res)

            num_iter += 1
        
        return preds, preds_meta

class NPointsPer2DSliceInteractive(twoDInteractivePrompter):
    def __init__(
            self,
            inferer: Inferer,
            seed: int = 11121,
            dof_bound: int | None = 60,
            perf_bound: float | None = 0.85,
            max_iter: int | None = 10,
            n_points_per_slice: int = 5
        ):
        super().__init__(inferer, seed, dof_bound, perf_bound, max_iter)
        self.n_points_per_slice = n_points_per_slice


    def get_initial_prompt_step(self) -> PromptStep:
        initial_prompt_step = get_pos_clicks2D_row_major(
            self.groundtruth, self.n_points_per_slice, self.seed
        )
        self.prompt_step_all = initial_prompt_step
        initial_dof = len(initial_prompt_step.get_slices_to_infer())*3 # 3 dofs per slice inferred

        return initial_prompt_step, initial_dof
    
interactive_prompt_styles = Literal[
    "NPointsPer2DSliceInteractive",
]