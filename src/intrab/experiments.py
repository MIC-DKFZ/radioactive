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

from intrab.utils.io import binarize_gt, verify_results_dir_exist
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
