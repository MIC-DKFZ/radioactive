from argparse import Namespace
import os
import warnings
import json

from utils.modelUtils import inferer_registry
from experiments import run_experiments


def get_img_gts_jhu(dataset_dir):
    images_dir = os.path.join(dataset_dir, 'imagesTr')
    labels_dir = os.path.join(dataset_dir, 'labelsTr')
    imgs_gts = [
        (os.path.join(images_dir, img_path), os.path.join(labels_dir, img_path.rstrip('_0000.nii.gz') + '.nii.gz'))
        for img_path in os.listdir(images_dir)  # Adjust the extension as needed
        if os.path.exists(os.path.join(labels_dir, img_path.rstrip('_0000.nii.gz') + '.nii.gz'))
    ]
    return(imgs_gts)

def get_imgs_gts_amos(dataset_dir):
    images_dir = os.path.join(dataset_dir, 'imagesTs')
    labels_dir = os.path.join(dataset_dir, 'labelsTs')
    imgs_gts = [
        (os.path.join(images_dir, img_path), os.path.join(labels_dir, os.path.basename(img_path)))
        for img_path in os.listdir(images_dir)  # Adjust the extension as needed
        if os.path.exists(os.path.join(labels_dir, os.path.basename(img_path)))
    ]
    return(imgs_gts)

checkpoint_registry = {
    'sam': '/home/t722s/Desktop/UniversalModels/TrainedModels/sam_vit_h_4b8939.pth',
    'sammed2d': '/home/t722s/Desktop/UniversalModels/TrainedModels/sam-med2d_b.pth'
}

dataset_registry={
    'abdomenAtlas': {'dir':'/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_2img/', 'dataset_func': get_img_gts_jhu}
}

if __name__ == '__main__':
    # Setup
    # warnings.filterwarnings('error')

    dataset_name = 'abdomenAtlas'
    model_name = 'sammed2d'
    results_dir = '/home/t722s/Desktop/ExperimentResults'
    

    exp_params = Namespace(
        n_click_random_points = 5,
        n_slice_point_interpolation = 5,
        n_slice_box_interpolation = 5,
        n_seed_points_point_propagation = 5, n_points_propagation = 5, 
    )
    device = 'cuda'
    seed = 11121
    label_overwrite = None
    experiment_overwrite = None

    supported_prompts = ['points', 'boxes', 'interactive']

    # label_overwrite = {
    #     # "background": 0,
    #     # "aorta": 1,
    #     # "gall_bladder": 2,
    #     # "kidney_left": 3,
    #     # "kidney_right": 4,
    #     # "liver": 5,
    #     # "pancreas": 6,
    #     "postcava": 7,
    #     "spleen": 8,
    #     # "stomach": 9
    # }

    

    # experiment_overwrite = ['box_propagation']    


    # Get (img path, gt path) pairs
    results_path = os.path.join(results_dir, model_name + '_' + dataset_name + '.json')
    dataset_func, dataset_dir = dataset_registry[dataset_name]['dataset_func'], dataset_registry[dataset_name]['dir']
    imgs_gts = dataset_func(dataset_dir)
    imgs_gts = imgs_gts[:1]

    # Get dataset dict if missing
    with open(os.path.join(dataset_dir, 'dataset.json'), 'r') as f:
        dataset_info = json.load(f)
    label_dict = dataset_info['labels']

    if label_overwrite:
        label_dict = label_overwrite

    # Load the model
    #inferer = SAMInferer(checkpoint_path, device)
    checkpoint_path = checkpoint_registry[model_name]
    inferer = inferer_registry[model_name](checkpoint_path, device)

    # Run experiments
    run_experiments(inferer, imgs_gts, results_path, label_dict,
                    exp_params, supported_prompts,
                    seed = 11121, experiment_overwrite = experiment_overwrite)
