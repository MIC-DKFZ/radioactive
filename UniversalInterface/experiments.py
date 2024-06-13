import os
import numpy as np
import json
from utils.modelUtils import load_sammed2d
from classes.SAMMed2DClass_new import SAMMed2DInferer
import utils.analysisUtils as anUt
import utils.promptUtils as prUt
from utils.imageUtils import read_reorient_nifti
from tqdm import tqdm

# Setup
device = 'cuda'
sammed2d_checkpoint_path = "/home/t722s/Desktop/UniversalModels/TrainedModels/sam-med2d_b.pth"
dataset_dir = '/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_2img/'
results_path = '/home/t722s/Desktop/ExperimentResults/sammed2d_abdomenatlasJHU_2img.json'
label_dict = {
        "background": 0,
        "aorta": 1,
        "stomach": 9
    }
seed = 11121
n_click_random_points = 5
n_slice_line_interpolation = 5
n_slice_box_interpolation = 5
n_seed_points_point_propagation, n_points_propagation = 5, 5
if label_dict is None:
    # Load dataset information including labels
    with open(os.path.join(dataset_dir, 'dataset.json'), 'r') as f:
        dataset_info = json.load(f)
    label_dict = dataset_info['labels']

images_dir = os.path.join(dataset_dir, 'imagesTr')
labels_dir = os.path.join(dataset_dir, 'labelsTr')

# Load the model
sammed2d_model = load_sammed2d(sammed2d_checkpoint_path, device)
sammed2d_inferer = SAMMed2DInferer(sammed2d_model)
sammed2d_inferer.verbose = False # No need for progress bars per inference

# Define experiments
experiment_names = [
    'random_points', 'bounding_boxes', 'bbox3d_sliced',
    'line_interpolation', 'box_interpolation',
    'point_propagation', 'box_propagation'
]

# Initialize results dictionary
results = {exp_name: {label: {} for label in label_dict if label != "background"} for exp_name in experiment_names}

# Loop through all image and label pairs
for filename in tqdm(os.listdir(images_dir), 'looping through files'):
    if filename.endswith(".nii.gz"):
        base_name = filename.rstrip('.nii.gz')
        img_path = os.path.join(images_dir, filename)
        gt_path = os.path.join(labels_dir, f"{base_name.rstrip('_0000')}.nii.gz")
        img, gt = read_reorient_nifti(img_path).astype(np.float32), read_reorient_nifti(gt_path).astype(int)

        # Loop through each organ label except the background
        for label_name, label_val in tqdm(label_dict.items(), desc = 'looping through organs\n', leave = False):
            if label_name == 'background':
                continue

            organ_mask = np.where(gt == label_val, 1, 0)
            if not np.any(organ_mask):  # Skip if no foreground for this label
                continue

            slices_to_infer = np.where(np.any(organ_mask, axis=(1, 2)))[0]
            if slices_to_infer.size == 0: # Skip if no fg
                continue

            experiments = {
                'random_points': lambda: prUt.get_pos_clicks2D_row_major(organ_mask, n_click_random_points, seed=seed),
                'bounding_boxes': lambda: prUt.get_minimal_boxes_row_major(organ_mask),
                'bbox3d_sliced': lambda: prUt.get_bbox3d_sliced(organ_mask),
                'line_interpolation': lambda: prUt.line_interpolation(prUt.get_fg_points_from_cc_centers(organ_mask, n_slice_line_interpolation)),
                'box_interpolation': lambda: prUt.box_interpolation(prUt.get_seed_boxes(organ_mask, n_slice_box_interpolation)),
                'point_propagation': lambda: prUt.point_propagation(sammed2d_inferer, img, prUt.get_seed_point(organ_mask, n_seed_points_point_propagation, seed), slices_to_infer, seed, n_points_propagation, verbose = False),
                'box_propagation': lambda: prUt.box_propagation(sammed2d_inferer, img, prUt.get_seed_box(organ_mask), slices_to_infer, verbose = False)
            }


            # Run experiments
            for exp_name, prompting_func in tqdm(experiments.items(), desc = 'looping through experiments\n', leave = False):
                prompt = prompting_func()
                if exp_name in ['point_propagation', 'box_propagation']: 
                    segmentation = prompt
                else:
                    segmentation = sammed2d_inferer.predict(img, prompt)
                dice_score = anUt.compute_dice(segmentation, organ_mask)
                results[exp_name][label_name][base_name] = dice_score
    
    sammed2d_inferer.clear_embeddings()

# Save results 
with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {results_path}")
