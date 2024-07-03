from utils.model import load_sam
from classes.SAMClass import SAMInferer
from argparse import Namespace
import os
import json

from experiments import run_experiments

if __name__ == '__main__':
    # Setup
    dataset_dir = '/home/t722s/Desktop/Datasets/amosForUniversegTest'
    results_path = '/home/t722s/Desktop/ExperimentResults/sam_amos.json'
    checkpoint_path = '/home/t722s/Desktop/UniversalModels/TrainedModels/sam_vit_h_4b8939.pth'
    label_dict = {
            "organ": 2
        }

    exp_params = Namespace(
        n_click_random_points = 5,
        n_slice_line_interpolation = 5,
        n_slice_box_interpolation = 5,
        n_seed_points_point_propagation = 5, n_points_propagation = 5, 
    )
    device = 'cuda'
    seed = 11121

    # Get (img path, gt path) pairs
    images_dir = os.path.join(dataset_dir, 'imagesTs')
    labels_dir = os.path.join(dataset_dir, 'labelsTs')
    imgs_gts = [
        (os.path.join(images_dir, img_path), os.path.join(labels_dir, os.path.basename(img_path)))
        for img_path in os.listdir(images_dir)  # Adjust the extension as needed
        if os.path.exists(os.path.join(labels_dir, os.path.basename(img_path)))
    ]

    imgs_gts = imgs_gts[:1]

    # Get dataset dict if missing
    if label_dict is None:
        # Load dataset information including labels
        with open(os.path.join(dataset_dir, 'dataset.json'), 'r') as f:
            dataset_info = json.load(f)
        label_dict = dataset_info['labels']

    # Load the model
    model = load_sam(checkpoint_path, device)
    inferer = SAMInferer(model)

    # Run experiments
    run_experiments(inferer, imgs_gts, results_path, label_dict,
                    exp_params,
                    seed = 11121)
