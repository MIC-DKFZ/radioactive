import os
import numpy as np
import json
import utils.analysis as anUt
import utils.prompt as prUt
from utils.image import read_reorient_nifti
from tqdm import tqdm
import warnings

def run_experiments(inferer, imgs_gts, results_path, label_dict,
                    exp_params, supported_prompts, pass_prev_prompts,
                    seed, experiment_overwrite = None):


    inferer.verbose = False # No need for progress bars per inference

    # Define experiments
    experiment_names = [
        'random_points', 'bounding_boxes', 'bbox3d_sliced',
        'point_interpolation', 'box_interpolation',
        'point_propagation', 'box_propagation'
    ]

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
            if slices_to_infer.size == 0: # Skip if no fg
                continue
            
            experiments = {}
            
            if 'points' in supported_prompts:
                experiments.update({
                    'random_points': lambda: prUt.get_pos_clicks2D_row_major(organ_mask, exp_params.n_click_random_points, seed=seed),
                    'point_interpolation': lambda: prUt.point_interpolation(prUt.get_fg_points_from_cc_centers(organ_mask, exp_params.n_slice_point_interpolation)),
                    'point_propagation': lambda: prUt.point_propagation(inferer, img, prUt.get_seed_point(organ_mask, exp_params.n_seed_points_point_propagation, seed), 
                                                                        slices_to_infer, seed, exp_params.n_points_propagation, verbose = False),
                })

                if 'interactive' in supported_prompts:
                    pass # Add interactive point experiments here
            
            if 'boxes' in supported_prompts:
                experiments.update({
                    'bounding_boxes': lambda: prUt.get_minimal_boxes_row_major(organ_mask),
                    'bbox3d_sliced': lambda: prUt.get_bbox3d_sliced(organ_mask),
                    'box_interpolation': lambda: prUt.box_interpolation(prUt.get_seed_boxes(organ_mask, exp_params.n_slice_box_interpolation)),
                    'box_propagation': lambda: prUt.box_propagation(inferer, img, prUt.get_seed_box(organ_mask), slices_to_infer, verbose = False)
                })
                
                if 'interactive' in supported_prompts:
                    pass # Add interactive box experiments here

            # Debugging: Overwrite experiments
            if experiment_overwrite:
                experiments = {ex: experiments[ex] for ex in experiment_overwrite if ex in experiments.keys()}

            # Run experiments
            ## Non-interactive experiments - same for all
            for exp_name, prompting_func in tqdm(experiments.items(), desc = 'looping through experiments', leave = False):
                if exp_name in ['point_propagation', 'box_propagation']: # Experiments with more than one forward pass
                    segmentation, prompt = prompting_func()
                else:
                    prompt = prompting_func()
                    segmentation = inferer.predict(img, prompt)
                dice_score = anUt.compute_dice(segmentation, organ_mask)
                results[exp_name][label_name][base_name] = dice_score

    
            inferer.clear_embeddings()

    # Save results 
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_path}")

