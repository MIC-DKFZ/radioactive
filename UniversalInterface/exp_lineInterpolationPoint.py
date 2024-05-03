import os
import json
from torch.cuda import empty_cache
import gc
import numpy as np
from tqdm.auto import tqdm

from utils.base_classes import Points
from utils.modelUtils import load_sam, load_sammed2d
from classes.SAMClass import SAMWrapper, SAMInferer
from classes.SAMMed2DClass import SAMMed2DInferer, SAMMed2DWrapper

import utils.promptUtils as prUt
import utils.analysisUtils as anUt
from utils.imageUtils import read_im_gt

# Load in models
def run_inference_for_inferer(inferer, image_paths, labels_paths, organ_labels_dict, n_points, seed):
    res_dict = {organ: {} for organ in organ_labels_dict.keys()}

    for img_path, gt_path in tqdm(zip(image_paths, labels_paths), desc = 'Inferring'):
        for organ, organ_label in organ_labels_dict.items():
            if organ_label == 0 or organ == 'background':
                continue
            
            # Load in image, prompt
            img, gt = read_im_gt(img_path, gt_path, organ_label)

            if np.all(gt==0):
                print(f'No foreground for {img_path} {organ}. Giving dice_score NaN')
                res_dict[organ][os.path.basename(img_path)] = np.nan
                continue

            simulated_clicks = prUt.get_fg_points_from_cc_centers(gt, n_points)
            coords = prUt.interpolate_points(simulated_clicks, kind = 'linear').astype(int)
            pts_prompt = Points({'coords': coords, 'labels': [1]*len(coords)})
            segmentation = inferer.predict(img, pts_prompt)
            
            # Segment and store results
            segmentation = inferer.predict(img, pts_prompt)
            dice_score = anUt.compute_dice(segmentation, gt)            

            res_dict[organ][os.path.basename(img_path)] = dice_score

    return(res_dict)

def run_explineinterpolationpoint(dataset_path, n_points, save_path, organ_labels_dict = None, seed = 11121, device = 'cuda'):
    # Get dataset directory info and initialise results variable
    if organ_labels_dict is None:
        with open(os.path.join(dataset_path, 'dataset.json'), 'r') as f:
            metadata = json.load(f)
            organ_labels_dict = metadata['labels']

    images_dir = os.path.join(dataset_path, 'imagesTr')
    labels_dir = os.path.join(dataset_path, 'labelsTr')
    image_paths = sorted(os.path.join(images_dir, f) for f in os.listdir(images_dir))
    labels_paths = sorted(os.path.join(labels_dir, f) for f in os.listdir(labels_dir))

    # Obtain SAM results
    sam_checkpoint_path = '/home/t722s/Desktop/UniversalModels/TrainedModels/sam_vit_h_4b8939.pth'
    sam_model = load_sam(sam_checkpoint_path, device)
    sam_wrapper = SAMWrapper(sam_model, device)
    sam_inferer = SAMInferer(sam_wrapper, device)

    print('Inferring with SAM')
    res_dict_sam = run_inference_for_inferer(sam_inferer, image_paths, labels_paths, organ_labels_dict, n_points, seed)

    # Clear memory
    del sam_inferer, sam_wrapper, sam_model
    empty_cache()
    gc.collect()

    # Obtain SAMMed2D results:
    print('Inferring with SAMMed2D')
    sammed2d_checkpoint_path = "/home/t722s/Desktop/UniversalModels/TrainedModels/sam-med2d_b.pth"
    sammed2d_model = load_sammed2d(sammed2d_checkpoint_path, device)
    sammed2d_wrapper = SAMMed2DWrapper(sammed2d_model, device)
    sammed2d_inferer = SAMMed2DInferer(sammed2d_wrapper)

    res_dict_sammed2d = run_inference_for_inferer(sammed2d_inferer, image_paths, labels_paths, organ_labels_dict, n_points, seed)

    # Write results
    res_dict = {'sam': res_dict_sam, 'sammed2d': res_dict_sammed2d, 'configs': {'dataset_path': dataset_path, 'n_points': n_points, 'seed': seed}}
    with open(save_path, 'w') as f:
        json.dump(res_dict, f, indent=4)

    print('Successfully finished inference')


if __name__ == '__main__':
    dataset_path = '/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_sub'
    organ_labels_dict = {
                        "aorta": 1,
                        "gall_bladder": 2,
                        "kidney_left": 3,
                        "liver": 5
                        }
    run_explineinterpolationpoint(dataset_path, 5, '/home/t722s/Desktop/inferenceResults/explineinterppoint5.json', organ_labels_dict, 11121)