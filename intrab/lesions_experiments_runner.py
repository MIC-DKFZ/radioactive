import argparse
import os
from datetime import datetime

from intrab.lesions_experiments_runner_segvol import sammed3d_lazy, segvol_lazy
from intrab.model.model_utils import medsam_lazy, sam_lazy, sammed2d_lazy
from lesions_experiments import run_experiments, run_postprocess



inferer_registry = {
    'sam': sam_lazy,
    'sammed2d': sammed2d_lazy,
    'medsam': medsam_lazy,
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
    'sam': '/home/t722s/Desktop/UniversalModels/TrainedModels/sam_vit_h_4b8939.pth',
    'medsam': '/home/t722s/Desktop/UniversalModels/TrainedModels/medsam_vit_b.pth',
    'sammed2d': '/home/t722s/Desktop/UniversalModels/TrainedModels/sam-med2d_b.pth',
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
    dataset_dir = '/home/t722s/Desktop/Datasets/melanoma_HD_sub/'
    #dataset_dir = '/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_2img'
    model_name = 'sammed2d'
    results_dir_all = '/home/t722s/Desktop/ExperimentResults_lesions'
    
    # results_dir = '/media/t722s/2.0 TB Hard Disk/lesions_experiments/'
    
    device = 'cuda'
    dataset_name = os.path.basename(dataset_dir.removesuffix('/'))

    # Get (img path, gt path) pairs
    results_dir = os.path.join(results_dir_all, dataset_name, model_name) # leave out time stamping  
    #results_dir = os.path.join(results_dir_all, dataset_name, model_name + '_' + datetime.now().strftime("%Y%m%d_%H%M"))
    imgs_gts = get_imgs_gts(dataset_dir)

    # Load the model
    checkpoint_path = checkpoint_registry[model_name]
    inferer = inferer_registry[model_name](checkpoint_path, device)

    # Run experiments
    exp_names = run_experiments(inferer, imgs_gts, dataset_dir, results_dir, target_is_lesion=True, save_segs = True)

    # TESTING
    # exp_names = ['bbox3d']
    # Merge instance segmentations and obtain merged dice
    run_postprocess(results_dir, exp_names, dataset_dir)
