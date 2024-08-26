# Experiments content
import os
from model.inferer import Inferer
import numpy as np
import json
import intrab.utils.analysis as analysis
from intrab.utils.interactivity import iterate_2d
from intrab.utils.image import read_reorient_nifti
from tqdm import tqdm
import warnings
import shutil


def run_experiments_2d(
    inferer: Inferer,
    imgs_gts,
    results_dir,
    label_dict,
    exp_params,
    prompt_types,
    seed,
    save_segs=False,
    experiment_overwrite=None,
):

    inferer.verbose = False  # No need for progress bars per inference

    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Define experiments
    experiments = {}

    if "points" in prompt_types:
        experiments.update(
            {
                "random_points": lambda organ_mask: prompt.get_pos_clicks2D_row_major(
                    organ_mask, exp_params.n_click_random_points, seed=seed
                ),
                "point_interpolation": lambda organ_mask: prompt.point_interpolation(
                    prompt.get_fg_points_from_cc_centers(organ_mask, exp_params.n_slice_point_interpolation)
                ),
                "point_propagation": lambda img, organ_mask, slices_to_infer: prompt.point_propagation(
                    inferer,
                    img,
                    prompt.get_seed_point(organ_mask, exp_params.n_seed_points_point_propagation, seed),
                    slices_to_infer,
                    seed,
                    exp_params.n_points_propagation,
                    verbose=False,
                ),
            }
        )

    if "boxes" in prompt_types:
        experiments.update(
            {
                "bounding_boxes": lambda organ_mask: prompt.get_minimal_boxes_row_major(organ_mask),
                "bbox3d_sliced": lambda organ_mask: prompt.get_bbox3d_sliced(organ_mask),
                "box_interpolation": lambda organ_mask: prompt.box_interpolation(
                    prompt.get_seed_boxes(organ_mask, exp_params.n_slice_box_interpolation)
                ),
                "box_propagation": lambda img, organ_mask, slices_to_infer: prompt.box_propagation(
                    inferer,
                    img,
                    prompt.get_seed_box(organ_mask),
                    slices_to_infer,
                    use_stored_embeddings=True,
                    verbose=False,
                ),
            }
        )

    interactive_experiments = {}
    if "interactive" in prompt_types:
        interactive_experiments.update(
            {
                "point_interpolation_interactive": lambda organ_mask: prompt.point_interpolation(
                    prompt.get_fg_points_from_cc_centers(organ_mask, exp_params.n_slice_point_interpolation)
                ),
                "point_propagation_interactive": lambda img, organ_mask, slices_to_infer: prompt.point_propagation(
                    inferer,
                    img,
                    prompt.get_seed_point(organ_mask, exp_params.n_seed_points_point_propagation, seed),
                    slices_to_infer,
                    seed,
                    exp_params.n_points_propagation,
                    use_stored_embeddings=True,
                    verbose=False,
                    return_low_res_logits=True,
                ),
            }
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
    results = {exp_name: {target: {} for target in targets} for exp_name in experiment_names}
    results["seed"] = seed

    # Loop through all image and label pairs
    for img_path, gt_path in tqdm(imgs_gts, desc="looping through files\n", leave=False):
        base_name = os.path.basename(gt_path)
        img, _ = read_reorient_nifti(img_path, np.float32)
        gt, inv_transform = read_reorient_nifti(gt_path, np.uint8)

        # Loop through each organ label except the background
        for target, target_label in tqdm(label_dict.items(), desc="looping through organs\n", leave=False):
            if target == "background":
                continue

            organ_mask = np.where(gt == target_label, 1, 0)
            if not np.any(organ_mask):  # Skip if no foreground for this label
                warnings.warn(f"{gt_path} missing segmentation for {target}")
                continue

            slices_to_infer = np.where(np.any(organ_mask, axis=(1, 2)))[0]

            # Handle non-interactive experiments
            for exp_name, prompting_func in tqdm(
                experiments.items(), desc="looping through non_interactive experiments", leave=False
            ):
                if exp_name in ["point_propagation", "box_propagation"]:
                    segmentation, prompt = prompting_func(img, organ_mask, slices_to_infer)
                else:
                    prompt = prompting_func(organ_mask)
                    segmentation = inferer.predict(img, prompt, use_stored_embeddings=True)
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
                if exp_name in ["point_propagation_interactive", "box_propagation_interactive"]:
                    segmentation, low_res_masks, prompt = prompting_func(img, organ_mask, slices_to_infer)
                    init_dof = 5
                else:
                    prompt = prompting_func(organ_mask)
                    segmentation, low_res_masks = inferer.predict(
                        img, prompt, return_low_res_logits=True, use_stored_embeddings=True
                    )
                    init_dof = 9

                if save_segs:
                    dice_scores, dofs, segmentations, prompts = iterate_2d(
                        inferer,
                        img,
                        organ_mask,
                        segmentation,
                        low_res_masks,
                        prompt,
                        inferer.pass_prev_prompts,
                        use_stored_embeddings=True,
                        scribble_length=0.6,
                        contour_distance=3,
                        disk_size_range=(0, 3),
                        init_dof=init_dof,
                        perf_bound=exp_params.perf_bound,
                        dof_bound=exp_params.dof_bound,
                        seed=seed,
                        verbose=False,
                        detailed=True,
                    )
                else:
                    dice_scores, dofs = iterate_2d(
                        inferer,
                        img,
                        organ_mask,
                        segmentation,
                        low_res_masks,
                        prompt,
                        inferer.pass_prev_prompts,
                        use_stored_embeddings=True,
                        scribble_length=0.6,
                        contour_distance=3,
                        disk_size_range=(0, 3),
                        init_dof=init_dof,
                        perf_bound=exp_params.perf_bound,
                        dof_bound=exp_params.dof_bound,
                        seed=seed,
                        verbose=False,
                    )

                results[exp_name][target][base_name] = {"dof": dofs, "dice_scores": dice_scores}

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
