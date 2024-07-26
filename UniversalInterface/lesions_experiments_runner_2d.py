import argparse
import os
from datetime import datetime

from lesions_experiments_2d import run_experiments_2d
from classes.SAMClass import SAMInferer
from classes.SAMMed2DClass import SAMMed2DInferer
from classes.MedSAMClass import MedSAMInferer
from classes.SAMMed3DClass import SAMMed3DInferer

inferer_registry = {
    'sam': SAMInferer,
    'sammed2d': SAMMed2DInferer,
    'medsam': MedSAMInferer,
    'sammed3d': SAMMed3DInferer
}
def get_imgs_gts(dataset_dir):
    imgs_gts = []
    for suffix in ['Tr', 'Ts']:
        images_dir = os.path.join(dataset_dir, 'images' + suffix)
        labels_dir = os.path.join(dataset_dir, 'labels' + suffix)
        imgs_gts.extend([
            (os.path.join(images_dir, img_path), os.path.join(labels_dir, img_path.removesuffix('_0000.nii.gz') + '.nii.gz'))
            for img_path in os.listdir(images_dir)  # Adjust the extension as needed
            # if os.path.exists(os.path.join(labels_dir, img_path.removesuffix('_0000.nii.gz') + '.nii.gz')) # Remove check. All the files should exist.
        ])

    return(imgs_gts)

def get_imgs_gts_sub(dataset_dir):
    imgs_gts = []
    for suffix in ['Tr']:
        images_dir = os.path.join(dataset_dir, 'images' + suffix)
        labels_dir = os.path.join(dataset_dir, 'labels' + suffix)
        imgs_gts.extend([
            (os.path.join(images_dir, img_path), os.path.join(labels_dir, img_path.removesuffix('_0000.nii.gz') + '.nii.gz'))
            for img_path in os.listdir(images_dir)  # Adjust the extension as needed
            # if os.path.exists(os.path.join(labels_dir, img_path.removesuffix('_0000.nii.gz') + '.nii.gz')) # Remove check. All the files should exist.
        ])

    return(imgs_gts)


checkpoint_registry = {
    'sam': '/home/t722s/Desktop/UniversalModels/TrainedModels/sam_vit_h_4b8939.pth',
    'medsam': '/home/t722s/Desktop/UniversalModels/TrainedModels/medsam_vit_b.pth',
    'sammed2d': '/home/t722s/Desktop/UniversalModels/TrainedModels/sam-med2d_b.pth'
}

dataset_registry={
    'infer_max_sub': {'dir':'/home/t722s/Desktop/Datasets/infer_max_sub/', 'dataset_func': get_imgs_gts_sub},
    'infer_max': {'dir':'/home/t722s/Desktop/Datasets/infer_max/', 'dataset_func': get_imgs_gts}
}

if __name__ == '__main__':
    # Setup
    # warnings.filterwarnings('error')

    parser = argparse.ArgumentParser(description='Example script to demonstrate argparse usage.')
    parser.add_argument('model_name', type = str, help = 'Select from "sam", "medsam", "sammed2d"')
    args = parser.parse_args()

    model_name = args.model_name
    # results_dir = '/home/t722s/Desktop/ExperimentResults_lesions'
    results_dir = '/media/t722s/2.0 TB Hard Disk/lesions_experiments/'

    dataset_name = 'infer_max'
    
    device = 'cuda'
    seed = 11121

    # Get (img path, gt path) pairs
    results_path = os.path.join(results_dir, model_name + '_' + dataset_name + '_' + datetime.now().strftime("%Y%m%d_%H%M"))
    dataset_func, dataset_dir = dataset_registry[dataset_name]['dataset_func'], dataset_registry[dataset_name]['dir']
    imgs_gts = dataset_func(dataset_dir)

    # Load the model
    checkpoint_path = checkpoint_registry[model_name]
    inferer = inferer_registry[model_name](checkpoint_path, device)

    # Run experiments
    run_experiments_2d(inferer, imgs_gts, results_path, save_segs = True)
