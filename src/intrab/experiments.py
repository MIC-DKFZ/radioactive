# Experiments content
from datetime import datetime
import os
from pathlib import Path
import pickle

from loguru import logger
from intrab.model.inferer import Inferer
import numpy as np
import json
from intrab.model.model_utils import get_wanted_supported_prompters
from intrab.prompts.prompt_hparams import PromptConfig
from intrab.prompts.prompter import static_prompt_styles

from intrab.prompts.prompter import Prompter
import nibabel as nib


from tqdm import tqdm
import shutil

from intrab.utils.io import binarize_gt, create_instance_gt, verify_results_dir_exist
from intrab.utils.io import verify_results_dir_exist
from nneval.evaluate_semantic import semantic_evaluation

from intrab.utils.result_data import PromptResult


# ToDo: Make this run_organ_experiments
def run_experiments(
    inferer: Inferer,
    imgs_gts: list[tuple[str, str]],
    results_dir: Path,
    label_dict: dict[str, int],
    pro_conf: PromptConfig,
    wanted_prompt_styles: list[static_prompt_styles],
    seed,
    experiment_overwrite=None,
    results_overwrite: bool = False,
    debug: bool = False,
):
    targets: dict = {k.replace("/", "_"): v for k, v in label_dict.items() if k != "background"}
    prompters: list[Prompter] = get_wanted_supported_prompters(inferer, pro_conf, wanted_prompt_styles, seed)
    verify_results_dir_exist(targets=targets, results_dir=results_dir, prompters=prompters)

    if debug:
        logger.warning("Debug mode activated. Only running on the first three images.")
        imgs_gts = imgs_gts[:3]

    logger.warning(
        "Coordinate systems should be checked and verified for correctness. \n"
        + "Right now this is assumed to be correct"
    )

    # Loop through all image and label pairs
    target_names: set[str] = set()
    for img_path, gt_path in tqdm(imgs_gts, desc="looping through files\n", leave=True):
        # Loop through each organ label except the background
        for target, target_label in tqdm(targets.items(), desc="Predicting targets...\n", leave=False):
            target_name: str = f"{target_label:03d}" + "__" + target
            target_names.add(target_name)

            # ---------------- Get binary gt in original coordinate system --------------- #
            base_name = os.path.basename(gt_path)
            bin_gt_filepath = results_dir / "binarised_gts" / target_name / base_name
            binary_gt_orig_coords = binarize_gt(gt_path, target_label)
            if not bin_gt_filepath.exists():
                binary_gt_orig_coords.to_filename(bin_gt_filepath)

            for prompter in tqdm(
                prompters,
                desc="Prompting with various prompters ...",
                leave=False,
                # disable=True,
            ):
                filepath = results_dir / prompter.name / target_name / base_name
                if filepath.exists() and not results_overwrite:
                    logger.debug(f"Skipping {gt_path} as it has already been processed.")
                    continue
                prompter.set_groundtruth(binary_gt_orig_coords)

                # Handle non-interactive experiments
                if prompter.is_static:
                    prediction_result: PromptResult
                    prediction_result = prompter.predict_image(image_path=img_path)
                    # Save the prediction
                    prediction_result.predicted_image.to_filename(filepath)
                # Handle interactive experiments
                else:
                    prediction_results: list[PromptResult]
                    prediction_results = prompter.predict_image(image_path=img_path)
                    base_path = filepath.parent

                    for cnt, pred in enumerate(prediction_results):
                        target_path = base_path / f"iter_{cnt}"
                        target_path.mkdir(exist_ok=True)
                        pred.predicted_image.to_filename(target_path / base_name)

                        # ToDo: Serialize the prediction results.

    # We always run the semantic eval on the created folders directly.
    for target_name in target_names:
        for prompter in prompters:
            if prompter.is_static:
                with logger.catch(level="WARNING"):
                    semantic_evaluation(
                        semantic_gt_path=results_dir / "binarised_gts" / target_name,
                        semantic_pd_path=results_dir / prompter.name / target_name,
                        output_path=results_dir / prompter.name,
                        classes_of_interest=(1,),
                        output_name=target_name,
                    )


def save_static_lesion_results(
    prompt_results: list[PromptResult], base_name: Path, semantic_filename: str, instance_filename: str
):
    """Iterate through all single prompt results that are"""
    semantic_pd_path = base_name / semantic_filename
    instance_pd_path = base_name / instance_filename
    for i, prompt_result in enumerate(prompt_results):
        if i == 0:
            img = prompt_result.predicted_image.get_fdata()
            affine = prompt_result.predicted_image.affine
            continue
        img = img + (prompt_result.predicted_image.get_fdata() + i)
    semantic_img = nib.Nifti1Image(np.where(np.nonzero(img), 1, 0), affine)
    instance_img = nib.Nifti1Image(img, affine)

    semantic_pd_path = base_name / semantic_filename
    instance_pd_path = base_name / instance_filename
    # ToDo: Save the prompt steps as well.
    instance_img.to_filename(instance_pd_path)
    semantic_img.to_filename(semantic_pd_path)


def save_interactive_lesion_results(
    all_prompt_results: list[list[PromptResult]], base_name: Path, semantic_filename: str, instance_filename: str
):
    """
    Save the interactive lesion results.
    Each lesion has it's own interactive results. The outer list count goes through the lesions and the inner list goes through the iterations.
    """
    all_arr = []
    affine = None
    for lesion_id, prompt_results in enumerate(all_prompt_results):
        all_arr_iter = []
        prompt_res: PromptResult
        for cnt, prompt_res in enumerate(prompt_results):
            if lesion_id == 0 and cnt == 0:
                affine - prompt_res.predicted_image.affine
            img = (prompt_res.predicted_image.get_fdata()) + lesion_id
            all_arr_iter.append(img)
        all_arr.append(all_arr_iter)
    joint_arr = np.array(all_arr)  # Shape: (num_lesions, num_iterations, x, y, z)
    iter_wise_arr = np.sum(joint_arr, axis=0, keepdims=False)  # Shape: (num_iterations, x, y, z)
    # Save the results iteratively
    for cnt in range(iter_wise_arr.shape[0]):
        semantic_img = nib.Nifti1Image(np.where(np.nonzero(iter_wise_arr[cnt]), 1, 0), affine)
        instance_img = nib.Nifti1Image(iter_wise_arr[cnt], affine)
        semantic_pd_path = base_name / f"iter_{cnt}" / semantic_filename
        instance_pd_path = base_name / f"iter_{cnt}" / instance_filename
        semantic_img.to_filename(semantic_pd_path)
        instance_img.to_filename(instance_pd_path)


def run_experiments_lesions(
    inferer: Inferer,
    imgs_gts: list[tuple[str, str]],
    results_dir: Path,
    label_dict: dict[str, int],
    pro_conf: PromptConfig,
    wanted_prompt_styles: list[static_prompt_styles],
    seed,
    experiment_overwrite=None,
    results_overwrite: bool = False,
    debug: bool = False,
):
    targets: dict = {k.replace("/", "_"): v for k, v in label_dict.items() if k != "background"}
    assert len(targets) == 1, "Lesion experiments only support two classes -- background and lesions"
    prompters: list[Prompter] = get_wanted_supported_prompters(inferer, pro_conf, wanted_prompt_styles, seed)
    verify_results_dir_exist(targets=targets, results_dir=results_dir, prompters=prompters)

    if debug:
        logger.warning("Debug mode activated. Only running on the first three images.")
        imgs_gts = imgs_gts[:3]

    logger.warning(
        "Coordinate systems should be checked and verified for correctness. \n"
        + "Right now this is assumed to be correct"
    )

    gt_output_path = results_dir / "instance_gts"
    pred_output_path = results_dir / "instance_pred"
    # Loop through all image and label pairs
    for img_path, gt_path in tqdm(imgs_gts, desc="looping through files\n", leave=True):
        # Loop through each organ label except the background
        filename = os.path.basename(gt_path)
        filename_wo_ext = ".".split(filename)[0]  # Remove the extension
        instance_filename = f"{filename_wo_ext}__ins__.nii.gz"
        semantic_filename = f"{filename_wo_ext}__sem__.nii.gz"
        semantic_gt_path = gt_output_path / semantic_filename
        instance_gt_path = gt_output_path / instance_filename

        # ---------------- Get binary gt in original coordinate system --------------- #
        base_name = os.path.basename(gt_path)
        bin_gt_filepath = pred_output_path / base_name
        semantic_nib, instance_nib, lesion_ids = create_instance_gt(gt_path)
        if semantic_gt_path.exists() and instance_gt_path.exists():
            semantic_nib.to_filename(semantic_gt_path)
            instance_nib.to_filename(instance_gt_path)

        prompter: Prompter
        for prompter in tqdm(
            prompters,
            desc="Prompting with various prompters ...",
            leave=False,
            # disable=True,
        ):
            base_name = pred_output_path / prompter.name
            all_prompt_result: list[PromptResult] | list[list[PromptResult]] = []
            for lesion_id in lesion_ids:
                binary_gt_orig_coords = binarize_gt(gt_path, lesion_id)
                prompter.set_groundtruth(binary_gt_orig_coords)

                # Handle non-interactive experiments
                if prompter.is_static:
                    semantic_pd_path = base_name / semantic_filename
                    instance_pd_path = base_name / instance_filename

                    if semantic_pd_path.exists() and instance_pd_path.exists() and not results_overwrite:
                        logger.debug(f"Skipping {gt_path} as it has already been processed.")
                        continue
                    prediction_result: PromptResult
                    prediction_result = prompter.predict_image(image_path=img_path)
                    # Save the prediction
                    all_prompt_result.append(prediction_result)
                # Handle interactive experiments
                else:
                    expected_sem_paths = [
                        base_name / f"iter_{i}" / semantic_filename for i in range(prompter.num_iterations)
                    ]
                    expected_ins_paths = [
                        base_name / f"iter_{i}" / instance_filename for i in range(prompter.num_iterations)
                    ]
                    if (
                        all([path.exists() for path in expected_sem_paths])
                        and all([path.exists() for path in expected_ins_paths])
                        and not results_overwrite
                    ):
                        logger.debug(f"Skipping {gt_path} as it has already been processed.")
                        continue
                    all_prompt_result.append(prompter.predict_image(image_path=img_path))

            # -------------------- Save static or interactive results -------------------- #
            if prompter.is_static:
                save_static_lesion_results(all_prompt_result, base_name, semantic_filename, instance_filename)
            else:
                save_interactive_lesion_results(all_prompt_result, base_name, semantic_filename, instance_filename)

    # We always run the semantic eval on the created folders directly.
    for target_name in target_names:
        for prompter in prompters:
            if prompter.is_static:
                with logger.catch(level="WARNING"):
                    semantic_evaluation(
                        semantic_gt_path=results_dir / "binarised_gts" / target_name,
                        semantic_pd_path=results_dir / prompter.name / target_name,
                        output_path=results_dir / prompter.name,
                        classes_of_interest=(1,),
                        output_name=target_name,
                    )
