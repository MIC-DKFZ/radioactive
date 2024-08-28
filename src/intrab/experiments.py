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

    # # Debugging: Overwrite experiments
    # if experiment_overwrite:
    #     experiments = {ex: experiments[ex] for ex in experiment_overwrite if ex in experiments.keys()}
    #     interactive_experiments = {
    #         ex: experiments[ex] for ex in experiment_overwrite if ex in interactive_experiments.keys()
    #     }
    # for p in prompters:
    #     results_dir / p.name
    # Todo: Remove when exclusion happens earlier.
    targets: dict = {k.replace("/", "_"): v for k, v in label_dict.items() if k != "background"}

    [
        Path(results_dir / p.name / target).mkdir(exist_ok=True, parents=True)
        for p in prompters
        for target in targets.keys()
    ]

    # Initialize results dictionary
    results = []

    # Loop through all image and label pairs
    for img_path, gt_path in tqdm(imgs_gts, desc="looping through files\n"):
        base_name = os.path.basename(gt_path)
        multi_class_gt = inferer.get_transformed_groundtruth(gt_path)

        # Loop through each organ label except the background
        for target, target_label in tqdm(targets.items(), desc="looping through organs\n"):
            binary_gt = np.where(multi_class_gt == target_label, 1, 0)

            if np.all(binary_gt == 0):
                logger.debug(f"Skipping {gt_path} missing segmentation for {target}")
                img = nib.load(gt_path)
                empty_gt = nib.Nifti1Image(binary_gt.astype(np.float32), img.affine)
                empty_gt.to_filename(results_dir / prompter.name / target / base_name)
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
                prediction, _ = prompter.predict_image(image_path=img_path)
                prediction.to_filename(results_dir / prompter.name / target / base_name)

            # # Now handle interactive experiments
            # for exp_name, prompting_func in tqdm(
            #     interactive_experiments.items(), desc="looping through interactive experiments", leave=False
            # ):
            #     # Set the few things that differ depending on the seed method
            #     if exp_name in ["point_propagation_interactive", "box_propagation_interactive"]:
            #         segmentation, low_res_masks, prompt = prompting_func(img, organ_mask, slices_to_infer)
            #         init_dof = 5
            #     else:
            #         prompt = prompting_func(organ_mask)
            #         segmentation, low_res_masks = inferer.predict(
            #             img, prompt, return_low_res_logits=True, use_stored_embeddings=True
            #         )
            #         init_dof = 9

            #     if save_segs:
            #         dice_scores, dofs, segmentations, prompts = iterate_2d(
            #             inferer,
            #             img,
            #             organ_mask,
            #             segmentation,
            #             low_res_masks,
            #             prompt,
            #             inferer.pass_prev_prompts,
            #             use_stored_embeddings=True,
            #             scribble_length=0.6,
            #             contour_distance=3,
            #             disk_size_range=(0, 3),
            #             init_dof=init_dof,
            #             perf_bound=exp_params.perf_bound,
            #             dof_bound=exp_params.dof_bound,
            #             seed=seed,
            #             verbose=False,
            #             detailed=True,
            #         )
            #     else:
            #         dice_scores, dofs = iterate_2d(
            #             inferer,
            #             img,
            #             organ_mask,
            #             segmentation,
            #             low_res_masks,
            #             prompt,
            #             inferer.pass_prev_prompts,
            #             use_stored_embeddings=True,
            #             scribble_length=0.6,
            #             contour_distance=3,
            #             disk_size_range=(0, 3),
            #             init_dof=init_dof,
            #             perf_bound=exp_params.perf_bound,
            #             dof_bound=exp_params.dof_bound,
            #             seed=seed,
            #             verbose=False,
            #         )

            #     results[exp_name][target][base_name] = {"dof": dofs, "dice_scores": dice_scores}

            #     if save_segs:
            #         for i, segmentation in enumerate(segmentations):
            #             seg_orig_ori = inv_transform(segmentation)  # Reorient segmentation
            #             save_path = os.path.join(results_dir, exp_name, target, base_name).replace(
            #                 ".nii.gz", f"_seg_{i}.nii.gz"
            #             )
            #             seg_orig_ori.to_filename(save_path)

            # inferer.clear_embeddings()

    # Save results

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_dir}")
