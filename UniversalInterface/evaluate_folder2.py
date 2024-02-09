import os
import argparse
import SimpleITK as sitk
from collections import defaultdict
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-rp', '--results_path', type=str)
parser.add_argument('-tdp', '--test_data_path', type=str)

args = parser.parse_args()

def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum


results_path = args.results_path
labels_dir = os.path.join(args.test_data_path, 'labelsTs')

with open(os.path.join(args.test_data_path, 'dataset.json'), 'r') as f:
    dataset_metadata = json.load(f)
labels_dict = dataset_metadata['labels']

eval_res = defaultdict(dict)

for label_name in os.listdir(labels_dir): # Loop thorugh ground truth label for images
    label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(labels_dir, label_name)))
    base_name = label_name.replace('.nii.gz','')
    
    for organ, label_num in labels_dict.items(): # loop through the scan's organs and their foreground label numbers for that scan
        label_num = int(label_num) # since it might be a string due to JSON
        label_binary = np.where(label == label_num, 1, np.zeros_like(label))
        organ_dir = os.path.join(args.results_path, organ)        

        seg_names = [file for file in os.listdir(organ_dir) if file.startswith(base_name + f'_pred_{organ}_trans')]

        organ_res = dict()
        for seg_name in seg_names: # Loop through the segmentations (ie for one click, two clicks etc.) for that image
            
            seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(organ_dir, seg_name)))
            seg_number = seg_name.removeprefix(base_name + f'_pred_{organ}_trans').removesuffix('.nii.gz') # Get segmentation number, ie number of clicks.
            dice = compute_dice(label, seg)
            organ_res[int(seg_number)+1] = round(dice, 4)
            eval_res[base_name][organ] = organ_res
            
with open(os.path.join(results_path, 'evaluation_dice.json'), 'w') as f:
    json.dump(eval_res, f, indent = 4)