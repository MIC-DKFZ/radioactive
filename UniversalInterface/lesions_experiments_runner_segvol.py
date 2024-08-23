import argparse
import os
from datetime import datetime

from lesions_experiments_segvol import run_experiments_3d

def segvol_lazy(checkpoint_path, device):
    from utils.class_segvol import SegVolInferer
    return SegVolInferer(checkpoint_path, device)

def sammed3d_lazy(checkpoint_path, device):
    from utils.class_SAMMed3D import SAMMed3DInferer
    return SAMMed3DInferer(checkpoint_path, device)

inferer_registry = {
    'segvol': segvol_lazy,
    'sammed3d': sammed3d_lazy
}

def get_imgs_gts(dataset_dir): 
    imgs_gts = {'Tr': [], 'Ts': []}
    for suffix in ['Tr', 'Ts']:
        images_dir = os.path.join(dataset_dir, 'images' + suffix)
        labels_dir = os.path.join(dataset_dir, 'labels' + suffix)
        if os.path.exists(images_dir):
            imgs_gts[suffix].extend([
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
    'segvol': '/home/t722s/Desktop/UniversalModels/TrainedModels/SegVol_v1.pth',
    'sammed3d': '/home/t722s/Desktop/UniversalModels/TrainedModels/sam_med3d_turbo.pth'
}

if __name__ == '__main__':
    # Setup
    # warnings.filterwarnings('error')

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model_name', type = str, required = True, help = 'Select from "sam", "medsam", "sammed2d"')
    # parser.add_argument('-d', '--dataset_dir', type = str, required = True, help = 'Path to dataset')
    # parser.add_argument('-r', '--results_dir', type = str, required = True, help = 'Path to desired results directory')
    # args = parser.parse_args()

    # model_name = args.model_name
    # dataset_dir = args.dataset_dir
    # results_dir = args.results_dir

    # Testing parameters:
    dataset_dir = '/home/t722s/Desktop/Datasets/Adrenal-ACC-Ki67_sub/'
    model_name = 'segvol'
    results_dir = '/home/t722s/Desktop/ExperimentResults_lesions_20240821'
    
    device = 'cuda'
    dataset_name = os.path.basename(dataset_dir.removesuffix('/'))

    # Get (img path, gt path) pairs
    results_path = os.path.join(results_dir, model_name + '_' + dataset_name) # leave out time stamping  + '_' + datetime.now().strftime("%Y%m%d_%H%M"))
    # results_path = os.path.join(results_dir, model_name + '_' + dataset_name + '_' + datetime.now().strftime("%Y%m%d_%H%M"))
    # imgs_gts = get_imgs_gts(dataset_dir)
    imgs_gts = {'Tr': [("/home/t722s/Desktop/Datasets/segvolTest/Case_image_00001_0000.nii.gz", "/home/t722s/Desktop/Datasets/segvolTest/Case_label_00001.nii.gz")],
                'Ts': []} # TESTING: provided test case

    # Load the model
    checkpoint_path = checkpoint_registry[model_name]
    inferer = inferer_registry[model_name](checkpoint_path, device)

    # Run experiments
    run_experiments_3d(inferer, imgs_gts, results_path, save_segs = True)
