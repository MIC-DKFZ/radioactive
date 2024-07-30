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
    'sammed2d': '/home/t722s/Desktop/UniversalModels/TrainedModels/sam-med2d_b.pth'
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
    model_name = 'sam'
    results_dir = '/home/t722s/Desktop/ExperimentResults_lesions'
    
    # results_dir = '/media/t722s/2.0 TB Hard Disk/lesions_experiments/'
    
    device = 'cuda'
    dataset_name = os.path.basename(dataset_dir.removesuffix('/'))

    # Get (img path, gt path) pairs
    results_path = os.path.join(results_dir, model_name + '_' + dataset_name) # leave out time stamping  + '_' + datetime.now().strftime("%Y%m%d_%H%M"))
    results_path = os.path.join(results_dir, model_name + '_' + dataset_name + '_' + datetime.now().strftime("%Y%m%d_%H%M"))
    imgs_gts = get_imgs_gts(dataset_dir)
    # imgs_gts = {'Tr': [('/home/t722s/Desktop/Datasets/melanoma_HD/imagesTr/0001996954_Follow-Up_2_0000.nii.gz', '/home/t722s/Desktop/Datasets/melanoma_HD/labelsTr/0001996954_Follow-Up_2.nii.gz')],
    #             'Ts': []} # TESTING: CUDA OUT OF MEMORY

    # Load the model
    checkpoint_path = checkpoint_registry[model_name]
    inferer = inferer_registry[model_name](checkpoint_path, device)

    # Run experiments
    run_experiments_2d(inferer, imgs_gts, results_path, save_segs = True)
