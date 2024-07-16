# Experiments content
import os
import numpy as np
import json
import utils.analysis as anUt
import utils.prompt as prUt
from utils.interactivity import iterate_2d
from utils.image import read_reorient_nifti
from tqdm import tqdm
import warnings

def run_experiments(inferer, imgs_gts, results_path, label_dict,
                    exp_params, prompt_types,
                    seed, experiment_overwrite = None):
    
    inferer.verbose = False # No need for progress bars per inference

    # Define experiments
    experiments = {}

    if 'points' in prompt_types:
        experiments.update({
            'random_points': lambda organ_mask: prUt.get_pos_clicks2D_row_major(organ_mask, exp_params.n_click_random_points, seed=seed),
            'point_interpolation': lambda organ_mask: prUt.point_interpolation(prUt.get_fg_points_from_cc_centers(organ_mask, exp_params.n_slice_point_interpolation)),
            'point_propagation': lambda organ_mask, slices_to_infer: prUt.point_propagation(inferer, img, prUt.get_seed_point(organ_mask, exp_params.n_seed_points_point_propagation, seed), 
                                                                slices_to_infer, seed, exp_params.n_points_propagation, verbose = False),
        })

    if 'boxes' in prompt_types:
        experiments.update({
            'bounding_boxes': lambda organ_mask: prUt.get_minimal_boxes_row_major(organ_mask),
            'bbox3d_sliced': lambda organ_mask: prUt.get_bbox3d_sliced(organ_mask),
            'box_interpolation': lambda organ_mask: prUt.box_interpolation(prUt.get_seed_boxes(organ_mask, exp_params.n_slice_box_interpolation)),
            'box_propagation': lambda organ_mask, slices_to_infer: prUt.box_propagation(inferer, img, prUt.get_seed_box(organ_mask), slices_to_infer, verbose = False)
        })

    interactive_experiments = {}
    if 'interactive' in prompt_types:
        interactive_experiments.update({
            'point_interpolation_interactive': lambda organ_mask: prUt.point_interpolation(prUt.get_fg_points_from_cc_centers(organ_mask, exp_params.n_slice_point_interpolation)),
            'point_propagation_interactive': lambda organ_mask, slices_to_infer: prUt.point_propagation(inferer, img, prUt.get_seed_point(organ_mask, exp_params.n_seed_points_point_propagation, seed), 
                                                                slices_to_infer, seed, exp_params.n_points_propagation, verbose = False, return_low_res_logits = True),
        })

    # Debugging: Overwrite experiments
    if experiment_overwrite:
        experiments = {ex: experiments[ex] for ex in experiment_overwrite if ex in experiments.keys()}
        interactive_experiments = {ex: experiments[ex] for ex in experiment_overwrite if ex in interactive_experiments.keys()}

    experiment_names = list(experiments.keys()) + list(interactive_experiments.keys())


    # Initialize results dictionary
    results = {exp_name: {label: {} for label in label_dict if label != "background"} for exp_name in experiment_names}

    # Loop through all image and label pairs
    #for filename in tqdm(os.listdir(images_dir), 'looping through files'):
    for img_path, gt_path in tqdm(imgs_gts, desc = 'looping through files\n'):
        base_name = os.path.basename(img_path)
        img, gt = read_reorient_nifti(img_path).astype(np.float32), read_reorient_nifti(gt_path).astype(int)

        # Loop through each organ label except the background
        for label_name, label_val in tqdm(label_dict.items(), desc = 'looping through organs\n', leave = False):
            if label_name == 'background':
                continue

            organ_mask = np.where(gt == label_val, 1, 0)
            if not np.any(organ_mask):  # Skip if no foreground for this label
                warnings.warn(f'{gt_path} missing segmentation for {label_name}')
                continue

            slices_to_infer = np.where(np.any(organ_mask, axis=(1, 2)))[0]
            
            # Handle non-interactive experiments
            for exp_name, prompting_func in tqdm(experiments.items(), desc = 'looping through non_interactive experiments', leave = False):
                if exp_name in ['point_propagation', 'box_propagation']: 
                    segmentation, prompt = prompting_func(organ_mask, slices_to_infer)
                else:
                    prompt = prompting_func(organ_mask)
                    segmentation = inferer.predict(img, prompt)
                dice_score = anUt.compute_dice(segmentation, organ_mask)
                results[exp_name][label_name][base_name] = dice_score

            # Now handle interactive experiments
            for exp_name, prompting_func in tqdm(interactive_experiments.items(), desc = 'looping through interactive experiments', leave = False):
                # Set the few things that differ depending on the seed method
                if exp_name in ['point_propagation_interactive', 'box_propagation_interactive']: 
                    segmentation, low_res_masks, prompt = prompting_func(organ_mask, slices_to_infer)
                    init_dof = 5
                else:
                    prompt = prompting_func(organ_mask)
                    segmentation, low_res_masks = inferer.predict(img, prompt, return_low_res_logits = True)
                    init_dof = 9
                
                dice_scores, dofs = iterate_2d(inferer, img, organ_mask, segmentation, low_res_masks, prompt, inferer.pass_prev_prompts,
                                                    scribble_length = 0.6, contour_distance = 3, disk_size_range= (0,3),
                                                    init_dof = init_dof, perf_bound = exp_params.perf_bound, dof_bound = exp_params.dof_bound, seed = seed, verbose = False)


                results[exp_name][label_name][base_name] = {'dof': dofs, 'dice_scores': dice_scores}               
                

            inferer.clear_embeddings()

    # Save results 
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_path}")
