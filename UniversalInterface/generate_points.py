import os
import argparse
import pickle
import json
from collections import defaultdict
import numpy as np
import SimpleITK as sitk
import tqdm
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-tdp', '--test_data_path', type=str)
parser.add_argument('-rp', '--results_path', type=str)
parser.add_argument('-nc', '--n_clicks', type=int, default=5)

args = parser.parse_args()

test_data_path = args.test_data_path
res_path = args.results_path
n_clicks = args.n_clicks

# Obtain test image and label paths. Only take those test images that have a label given
labels_dir = os.path.join(test_data_path, 'labelsTs')
test_label_paths = [os.path.join(labels_dir, label) for label in os.listdir(labels_dir)]
test_image_paths = [label_path.replace('labels', 'images') for label_path in test_label_paths]

# Obtain labels dictionary from dataset metadata
dataset_metadata_file = os.path.join(test_data_path, 'dataset.json')

with open(dataset_metadata_file, 'r') as f:
    dataset_metadata = json.load(f)

labels_dict = dataset_metadata['labels']
fg_labels_dict = {k:int(v) for k, v in labels_dict.items() if int(v) != 0} # Reverse order since it's name -> int in the dataset.json

# Create (point) prompts to use for each image and each fg label. 
# Two types: 2D: n_clicks points sampled uniformly at random per slice with foreground, 0 clicks per slice with no foreground; 3D: n_clicks sampled uniformly at random from the foreground region of the volume
full_prompt_dict = {}

for gt_mask_path in tqdm(test_label_paths):
    gt_mask = sitk.GetArrayFromImage(sitk.ReadImage(gt_mask_path))
    prompt_dict = defaultdict(dict) # For storing the prompts (per label and for 2D and 3D) for this particular image

    # proceed through labels
    for organ, label in tqdm(fg_labels_dict.items()):
        volume_fg = np.where(gt_mask == label) # Get foreground indices as three lists
        volume_fg = tuple(arr.astype(int) for arr in volume_fg) 

        n_fg_voxels = len(volume_fg[0])

        if n_fg_voxels == 0:
            tqdm.write(f'WARNING: Mask {gt_mask_path} is missing a segmentation for {organ}. An empty prompt list will be supplied')
            prompt_dict[label]['3D'] = np.array([])
            prompt_dict[label]['2D'] = np.array([])
            continue

        fg_slices = np.unique(volume_fg[0]) # Obtain superior axis slices which have foreground before reformating indices


        # 3D point generation:
        try: 
            point_indices = np.random.choice(len(volume_fg[0]), size = n_clicks, replace = False)
        except ValueError:
            raise RuntimeError(f'More points were requested than the number of foreground pixels in the volume ({n_clicks} vs {n_fg_voxels})')


        points3D = [(volume_fg[0][idx], volume_fg[1][idx], volume_fg[2][idx]) for idx in point_indices] # change from triple of arrays format to list of triples format
        prompt_dict[label]['3D'] = np.array(points3D)


        # 2D point generation:
        points2D = []
        warning_zs = {} # tracks slices without enough foreground, if any should exist

        for slice_index in fg_slices:
            slice_fg = np.where(gt_mask[slice_index,:,:] == label)
            slice_fg = tuple(arr.astype(int) for arr in slice_fg) 

            n_fg_pixels = len(slice_fg[0])
            if n_fg_pixels >= n_clicks:
                point_indices = np.random.choice(n_fg_pixels, size = n_clicks, replace = False)
            else:
                # In this case, take all foreground pixels and then obtain some duplicate points by sampling with replacement additionally
                warning_zs[f'z = {slice_index}'] = n_fg_pixels
                point_indices = np.concatenate([np.arange(n_fg_pixels),
                                            np.random.choice(n_fg_pixels, size = n_clicks-n_fg_pixels, replace = True)])
            
            points2D.extend([(slice_index, slice_fg[0][idx], slice_fg[1][idx]) for idx in point_indices])

        if warning_zs:
            tqdm.write(f'WARNING: some slices in {gt_mask_path} had fewer than n_clicks = {n_clicks} foreground pixels. Specifically: {warning_zs}')

        prompt_dict[label]['2D'] = np.array(points2D)

    full_prompt_dict[os.path.basename(gt_mask_path)] = dict(prompt_dict)

print(f'Saving prompts to {os.path.join(res_path, "prompts.pkl")}')
os.makedirs(res_path, exist_ok = True)
with open(os.path.join(res_path, 'prompts.pkl'), 'wb') as f:
    pickle.dump(full_prompt_dict, f)
