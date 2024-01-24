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

eval_res = defaultdict(dict)

for label_name in os.listdir(labels_dir):
    label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(labels_dir, label_name)))
    base_name = label_name.replace('.nii.gz','')
    seg_names = [file for file in os.listdir(results_path) if file.startswith(base_name + '_pred_trans')]

    for seg_name in seg_names:
        seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(results_path, seg_name)))
        dice = compute_dice(label, seg)
        eval_res[label_name][seg_name] = dice
with open(os.path.join(results_path, 'evaluation_dice.json'), 'w') as f:
    json.dump(eval_res, f, indent = 4)