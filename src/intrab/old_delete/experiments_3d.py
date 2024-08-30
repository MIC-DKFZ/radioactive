# Experiments content
import os
from model.inferer import Inferer
import numpy as np
import json
import intrab.utils.analysis as analysis
import intrab.prompts.prompt_3d as prUt
from intrab.utils.interactivity import iterate_3d
from intrab.utils.image import read_reorient_nifti
from tqdm import tqdm
import warnings


def run_experiments_3d(
    inferer: Inferer,
    imgs_gts,
    results_dir,
    label_dict,
    exp_params,
    prompt_types,
    seed,
    experiment_overwrite=None,
    save_segs=False,
):

    inferer.verbose = False  # No need for progress bars per inference

    # Define experiments
    experiments = {}

    if "points" in prompt_types:
        experiments.update(
            {
                "random_points": lambda organ_mask: prUt.get_pos_clicks3D(
                    organ_mask, exp_params.n_click_random_points, seed=seed
                )
            }
        )

    if "boxes" in prompt_types:
        experiments.update(
            {
                "bbox3d": lambda organ_mask: prUt.get_bbox3d(organ_mask),
            }
        )

    interactive_experiments = {}
    if "interactive" in prompt_types:
        interactive_experiments.update(
            {"points_interactive": lambda organ_mask: prUt.get_pos_clicks3D(organ_mask, 1, seed=seed)}
        )

    # Debugging: Overwrite experiments
    if experiment_overwrite:
        experiments = {ex: experiments[ex] for ex in experiment_overwrite if ex in experiments.keys()}
        interactive_experiments = {
            ex: experiments[ex] for ex in experiment_overwrite if ex in interactive_experiments.keys()
        }

    experiment_names = list(experiments.keys()) + list(interactive_experiments.keys())
    targets = [label for label in label_dict if label != "background"]

    if save_segs:
        dir_list = [
            os.path.join(results_dir, exp_name, target) for exp_name in experiment_names for target in targets
        ]
        for dir in dir_list:
            os.makedirs(dir, exist_ok=True)

    # Initialize results dictionary
    results = {
        exp_name: {label: {} for label in label_dict if label != "background"} for exp_name in experiment_names
    }

    # Loop through all image and label pairs
    # for filename in tqdm(os.listdir(images_dir), 'looping through files'):
    for img_path, gt_path in tqdm(imgs_gts, desc="looping through files\n"):
        base_name = os.path.basename(img_path)
        img, _ = read_reorient_nifti(img_path, np.float32, RAS=True)
        gt, inv_transform = read_reorient_nifti(gt_path, np.uint8, RAS=True)

        # Loop through each organ label except the background
        for target, target_label in tqdm(label_dict.items(), desc="looping through organs\n", leave=False):
            if target == "background":
                continue

            organ_mask = np.where(gt == target_label, 1, 0)
            if not np.any(organ_mask):  # Skip if no foreground for this label
                warnings.warn(f"{gt_path} missing segmentation for {target}")
                continue

            # Handle non-interactive experiments
            for exp_name, prompting_func in tqdm(
                experiments.items(), desc="looping through non_interactive experiments", leave=False
            ):
                prompt = prompting_func(organ_mask)
                segmentation = inferer.predict(img, prompt)
                dice_score = analysis.compute_dice(segmentation, organ_mask)
                results[exp_name][target][base_name] = dice_score

                if save_segs:
                    seg_orig_ori = inv_transform(segmentation)  # Reorient segmentation
                    save_path = os.path.join(
                        results_dir, exp_name, target, os.path.basename(gt_path).replace(".nii.gz", "_seg.nii.gz")
                    )
                    seg_orig_ori.to_filename(save_path)

            # Now handle interactive experiments
            for exp_name, prompting_func in tqdm(
                interactive_experiments.items(), desc="looping through interactive experiments", leave=False
            ):
                # Set the few things that differ depending on the seed method

                if save_segs:
                    dice_scores, dofs = iterate_3d(
                        inferer, img, gt, inferer.pass_prev_prompts, exp_params.perf_bound, exp_params.dof_bound, seed
                    )
                else:
                    dice_scores, dofs, segmentations, prompts = iterate_3d(
                        inferer,
                        img,
                        gt,
                        inferer.pass_prev_prompts,
                        exp_params.perf_bound,
                        exp_params.dof_bound,
                        seed,
                        detailed=True,
                    )
                results[exp_name][target_label][base_name] = {"dof": dofs, "dice_scores": dice_scores}

                if save_segs:
                    for i, segmentation in enumerate(segmentations):
                        seg_orig_ori = inv_transform(segmentation)  # Reorient segmentation
                        save_path = os.path.join(results_dir, exp_name, target, base_name).replace(
                            ".nii.gz", f"_seg_{i}.nii.gz"
                        )
                        seg_orig_ori.to_filename(save_path)

            inferer.clear_embeddings()

    # Save results
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_dir}")
