# Experiments content
from datetime import datetime
import os
from pathlib import Path

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

from intrab.utils.io import binarize_gt
from nneval.evaluate_semantic import semantic_evaluation


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
    if debug:
        logger.warning("Debug mode activated. Only running on the first three images.")
        imgs_gts = imgs_gts[:3]

    results_dir: Path
    if os.path.exists(results_dir):
        if results_overwrite:
            shutil.rmtree(results_dir)
        else:
            results_dir = results_dir.parent / (results_dir.name + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            # raise FileExistsError("Results directory already exists. Set results_overwrite=True to overwrite.")

    results_dir.mkdir(parents=True)

    # Define experiments
    prompters: list[Prompter] = get_wanted_supported_prompters(inferer, pro_conf, wanted_prompt_styles, seed)

    logger.warning(
        "Coordinate systems should be checked and verified for correctness. \n"
        + "Right now this is assumed to be correct"
    )

    targets: dict = {k.replace("/", "_"): v for k, v in label_dict.items() if k != "background"}

    [
        Path(results_dir / p.name / (f"{target_label:03d}" + "__" + target)).mkdir(exist_ok=True, parents=True)
        for p in prompters
        for target, target_label in targets.items()
    ]

    [
        Path(results_dir / "binarised_gts" / (f"{target_label:03d}" + "__" + target)).mkdir(
            exist_ok=True, parents=True
        )
        for target, target_label in targets.items()
    ]

    # Initialize results dictionary

    # Loop through all image and label pairs
    target_names: set[str] = set()
    for img_path, gt_path in tqdm(imgs_gts, desc="looping through files\n", leave=True):
        base_name = os.path.basename(gt_path)
        multi_class_gt = inferer.get_transformed_groundtruth(gt_path)

        # Loop through each organ label except the background
        for target, target_label in tqdm(targets.items(), desc="Predicting targets...\n", leave=False):
            target_name: str = f"{target_label:03d}" + "__" + target
            target_names.add(target_name)
            binary_gt = np.where(multi_class_gt == target_label, 1, 0)
            # Save the binarised ground truth next to the predictions for easy access -- Needed for evaluation
            binarize_gt(gt_path, target_label).to_filename(results_dir / "binarised_gts" / target_name / base_name)

            if np.all(binary_gt == 0):
                logger.debug(f"Skipping {gt_path} missing segmentation for {target}")
                img = nib.load(gt_path)
                empty_gt = nib.Nifti1Image(binary_gt.astype(np.float32), img.affine)
                empty_gt.to_filename(results_dir / prompter.name / target_name / base_name)
                continue

            # ToDo: Include again, Just temporary measure to see if inference works.
            # if not np.any(binary_gt):  # Skip if no foreground for this label
            #     logger.warning(f"{gt_path} missing segmentation for {target}")
            #     continue

            # Handle non-interactive experiments
            for prompter in tqdm(
                prompters,
                desc="Prompting with various prompters ...",
                leave=False,
                # disable=True,
            ):
                prompter.set_groundtruth(binary_gt)
                if prompter.is_static:
                    prediction, _ = prompter.predict_image(image_path=img_path)
                    prediction.to_filename(results_dir / prompter.name / target_name / base_name)
                else:
                    # do something else
                    pass

    # We always run the semantic eval on the created folders directly.
    for target_name in target_names:
        for prompter in prompters:
            if prompter.is_static:
                with logger.catch(level="WARNING"):
                    semantic_evaluation(
                        semantic_pd_path=results_dir / "binarised_gts" / target_name,
                        semantic_gt_path=results_dir / prompter.name / target_name,
                        output_path=results_dir / prompter.name,
                        classes_of_interest=(1,),
                        output_name=target_name,
                    )
