import cProfile
import pstats
import os
import numpy as np
import json
from utils.modelUtils import load_sammed2d
from Project.UniversalInterface.classes.SAMMed2DClass import SAMMed2DInferer
import utils.analysisUtils as anUt
import utils.promptUtils as prUt
from utils.imageUtils import read_reorient_nifti



# Setup
device = 'cuda'
sammed2d_checkpoint_path = "/home/t722s/Desktop/UniversalModels/TrainedModels/sam-med2d_b.pth"
dataset_dir = '/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_2img/'
results_path = '/home/t722s/Desktop/ExperimentResults/sammed2d_abdomenatlasJHU_2img.json'
label_dict = {
        "kidney_left": 3
    }

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

# Define experiments
experiment_names = [
    'random_points', 'bounding_boxes', 'bbox3d_sliced',
    'line_interpolation', 'box_interpolation',
    'point_propagation', 'box_propagation'
]

# Initialize results dictionary
results = {exp_name: {label: {} for label in label_dict if label != "background"} for exp_name in experiment_names}

# Loop through all image and label pairs
for filename in os.listdir(images_dir)[:1]:
    if filename.endswith(".nii.gz"):
        base_name = filename.rstrip('.nii.gz')
        img_path = os.path.join(images_dir, filename)
        gt_path = os.path.join(labels_dir, f"{base_name.rstrip('_0000')}.nii.gz")
        img, gt = read_reorient_nifti(img_path).astype(np.float32), read_reorient_nifti(gt_path).astype(int)

        # Loop through each organ label except the background
        for label_name, label_val in list(label_dict.items()):
            if label_name == 'background':
                continue

            organ_mask = np.where(gt == label_val, 1, 0)
            if not np.any(organ_mask):  # Skip if no foreground for this label
                continue

            slices_to_infer = np.where(np.any(organ_mask, axis=(1, 2)))[0]
            if slices_to_infer.size == 0: # Skip if 
                continue

            experiments = [
                ('random_points', lambda: prUt.get_pos_clicks2D_row_major(organ_mask, 5, seed=11121)),
                ('bounding_boxes', lambda: prUt.get_minimal_boxes_row_major(organ_mask, 3, 3)),
                ('bbox3d_sliced', lambda: prUt.get_bbox3d_sliced(organ_mask)),
                ('line_interpolation', lambda: prUt.line_interpolation(prUt.get_fg_points_from_cc_centers(organ_mask, 5))),
                ('box_interpolation', lambda: prUt.box_interpolation(prUt.get_seed_boxes(organ_mask, 5))),
                ('point_propagation', lambda: prUt.point_propagation(sammed2d_inferer, img, prUt.get_seed_point(organ_mask, 5), slices_to_infer, 11121, 5)),
                ('box_propagation', lambda: prUt.box_propagation(sammed2d_inferer, img, prUt.get_seed_box(organ_mask), slices_to_infer))
            ]


            one_call_prompt = experiments[0][1]
            many_calls_func = experiments[-1][1]

            prompt = one_call_prompt()
            sammed2d_inferer.predict(img, prompt)

            benchmark_folder = '/home/t722s/Desktop/benchmarking/'
            profiler = cProfile.Profile()
            profiler.enable()

            # Run your one-call function
            prompt = one_call_prompt()
            sammed2d_inferer.predict(img, prompt)

            profiler.disable()

            # Save the stats to a file
            one_call_stats_file = os.path.join(benchmark_folder, 'one_call_func_stats.prof')  # Define the filename
            stats = pstats.Stats(profiler)
            stats.dump_stats(one_call_stats_file)  # This saves the raw stats data
            stats.sort_stats('cumulative')
            with open(os.path.join(benchmark_folder, 'one_call_func_stats.txt'), 'w') as f:
                stats.stream = f
                stats.print_stats()  # This writes the formatted stats to the file

            profiler = cProfile.Profile()
            profiler.enable()

            # Run your many-calls function
            many_calls_func()

            profiler.disable()

            # Save the stats to a file
            many_calls_stats_file = os.path.join(benchmark_folder, 'many_calls_func_stats.prof')  # Define the filename
            stats = pstats.Stats(profiler)
            stats.dump_stats(many_calls_stats_file)  # This saves the raw stats data
            stats.sort_stats('cumulative')
            with open(os.path.join(benchmark_folder,'many_calls_func_stats.txt'), 'w') as f:
                stats.stream = f
                stats.print_stats()  # This writes the formatted stats to the file

