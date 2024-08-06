# Experiments content
import os
import numpy as np
import json
import utils.analysis as anUt
import utils.prompt as prUt
from utils.prompt import get_bbox3d
from utils.prompt_3d import get_pos_clicks3D
from utils.image import read_reorient_nifti
from tqdm import tqdm
import nibabel as nib
import shutil

def run_experiments_3d(inferer, imgs_gts, results_dir, save_segs = False):
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Define experiments
    experiments = {}

    experiments.update({
        'bbox3d': lambda organ_mask: get_bbox3d(organ_mask)
    })

    if save_segs:
        dir_list = [os.path.join(results_dir, 'segmentations'+suffix, os.path.basename(gt).removesuffix('.nii.gz')) 
                    for suffix, img_gts_sub in imgs_gts.items() 
                    for img, gt in img_gts_sub]
        for dir in dir_list:
            os.makedirs(dir, exist_ok = True)


    # Initialize results dictionary
    results = {suffix: {os.path.basename(gt): {} 
                    for img, gt in imgs_gts_sub} 
           for suffix, imgs_gts_sub in imgs_gts.items()}

    status = 'All segmentations successfully generated'

    # Loop through all image and label pairs
    for suffix in ['Tr', 'Ts']:
        for img_path, gt_path in tqdm(imgs_gts[suffix], desc = 'looping through files\n', leave = False):
            base_name = os.path.basename(gt_path)
            gt, inv_transform = read_reorient_nifti(gt_path, np.uint8, RAS = True)

            instances_present = np.unique(gt)
            instances_present = instances_present[instances_present!=0] # remove background

            # Loop through each instance of lesion present
            for instance in tqdm(instances_present, leave = False):
                organ_mask = np.where(gt == instance, 1, 0)

                # Handle experiment. Kept in a loop despite being one item for simplicity of adapting code from original
                for exp_name, prompting_func in experiments.items():
                    prompt = prompting_func(organ_mask)
                    try:
                        segmentation = inferer.predict(img_path, prompt, 'box')
                        dice_score = anUt.compute_dice(segmentation, organ_mask)

                        if save_segs:
                            seg_nifti = nib.Nifti1Image(segmentation, affine = np.eye(4))
                            save_path = os.path.join(results_dir, 'segmentations' + suffix, base_name.removesuffix('.nii.gz'), 'instance_' + str(instance) + '_seg.nii.gz')
                            seg_nifti.to_filename(save_path)
                    except: 
                        dice_score = None
                        status = 'Some segmentations failed'
                    results[suffix][base_name][str(instance)] = dice_score

    # Save results 
    results_path = os.path.join(results_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_dir}; {status}")
