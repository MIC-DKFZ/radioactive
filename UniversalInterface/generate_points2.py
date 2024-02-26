import os
import argparse
import pickle
import json
from collections import defaultdict
import numpy as np
import SimpleITK as sitk
import tqdm
from tqdm.auto import tqdm
from skimage.morphology import dilation, ball, disk

parser = argparse.ArgumentParser()
parser.add_argument('-tdp', '--test_data_path', type=str)
parser.add_argument('-rp', '--results_path', type=str)
parser.add_argument('-fg', '--n_fg_clicks', type=int, default=5)
parser.add_argument('-bg', '--n_bg_clicks', type=int, default=0)
parser.add_argument('-d', '--border_distance', type=int, default=3)
parser.add_argument('--seed', type=int)

args = parser.parse_args()

test_data_path = args.test_data_path
res_path = args.results_path
n_fg_clicks = args.n_fg_clicks
n_bg_clicks = args.n_bg_clicks
border_distance = args.border_distance
seed = args.seed


from skimage.morphology import dilation, ball, disk

def joinShuffleClicks(pos_clicks, neg_clicks):
    n_fg_clicks, n_bg_clicks = len(pos_clicks), len(neg_clicks)

    clicks = np.concatenate([pos_clicks, neg_clicks], axis = 0)
    labels = np.array([1]*n_fg_clicks + [0]*n_bg_clicks, dtype = int)
    shuffle_inds = np.random.permutation(n_fg_clicks+n_bg_clicks - 1)
    clicks[1:], labels[1:] = clicks[1:][shuffle_inds], labels[1:][shuffle_inds]

    return(clicks, labels)

if seed is not None:
    print(f'Set seed {seed}')
    np.random.seed(seed)

# Obtain test image and label paths. Only take those test images that have a label given
labels_dir = os.path.join(test_data_path, 'labelsTs')
test_label_paths = [os.path.join(labels_dir, label) for label in os.listdir(labels_dir)]
test_image_paths = [label_path.replace('labels', 'images') for label_path in test_label_paths]


# Obtain labels dictionary from dataset metadata
dataset_metadata_file = os.path.join(test_data_path, 'dataset.json')

with open(dataset_metadata_file, 'r') as f:
    dataset_metadata = json.load(f)

labels_dict = dataset_metadata['labels']

try: 
    fg_labels_dict = {k:int(v) for k, v in labels_dict.items() if k != 'background'} # apply int(v) since it's often serialised as a string

except ValueError:
    raise RuntimeError("Couldn't extract foreground label numbers; is the dataset json labels entry not in the expected name -> number format?")


# Create (point) prompts to use for each image and each fg label. 
# Two types: 2D: n_fg_clicks points sampled uniformly at random per slice with foreground, 0 clicks per slice with no foreground; 3D: n_fg_clicks sampled uniformly at random from the foreground region of the volume
full_prompt_dict = {}

for gt_mask_path in tqdm(test_label_paths):
    gt_mask = sitk.GetArrayFromImage(sitk.ReadImage(gt_mask_path))
    prompt_dict = defaultdict(dict) # For storing the prompts (per label and for 2D and 3D) for this particular image

    # proceed through labels
    for organ, label in tqdm(fg_labels_dict.items()):
        gt_organ = np.where(gt_mask ==label, 1, np.zeros_like(gt_mask))

        # 3D Point generation
        volume_fg = np.where(gt_organ) # Get foreground indices (formatted as triple of arrays)
        volume_fg = np.array(volume_fg).T # Reformat to numpy array of shape n_fg_voxels x 3

        n_fg_voxels = volume_fg.shape[0]

        if n_fg_voxels == 0:
            tqdm.write(f'WARNING: Mask {gt_mask_path} is missing a segmentation for {organ}. An empty prompt list will be supplied')
            prompt_dict[label]['3D'] = np.array([])
            prompt_dict[label]['2D'] = np.array([])
            continue

        ## Foreground points
        if n_fg_voxels < n_fg_clicks:
            raise RuntimeError(f'More foreground points were requested than the number of foreground xoxels in the volume ({n_fg_clicks} vs {n_fg_voxels})')

        point_indices = np.random.choice(n_fg_voxels, size = n_fg_clicks, replace = False)
        pos_clicks = volume_fg[point_indices]  # change from triple of arrays format to list of triples format

        ## Background points
        struct_element = ball(border_distance)
        volume_dilated = dilation(gt_organ, struct_element)
        border_region = volume_dilated - gt_organ

        volume_border = np.where(border_region)
        volume_border = np.array(volume_border).T
        n_border_voxels = volume_border.shape[0]

        if n_border_voxels < n_bg_clicks:
            raise RuntimeError(f'More background points were requested than the number of border voxels in the volume ({n_bg_clicks} vs {n_border_voxels})')
        
        point_indices = np.random.choice(n_border_voxels, size = n_bg_clicks, replace = False)
        neg_clicks = volume_border[point_indices]  # change from triple of arrays format to list of triples format # change from triple of arrays format to list of triples format

        ## Join together the clicks and labels. Shuffle all but the first point, which must always be foreground.
        clicks, labels = joinShuffleClicks(pos_clicks, neg_clicks) 

        clicks2, labels2 = joinShuffleClicks(pos_clicks, neg_clicks)

        prompt_dict[label]['3D'] = (clicks, labels)


        # 2D point generation:
        fg_slices = np.unique(volume_fg[:,0]) # Obtain superior axis slices which have foreground before reformating indices

        points2D = np.empty(shape = (0,3), dtype = int)
        labels2D = np.empty(shape = 0, dtype = int)
        warning_zs = {} # track slices without enough foreground/border, if any should exist

        for slice_index in fg_slices:
            ## Foreground points
            slice = gt_organ[slice_index]
            slice_fg = np.where(slice)
            slice_fg = np.array(slice_fg).T

            n_fg_pixels = slice_fg.shape[0]
            if n_fg_pixels >= n_fg_clicks:
                point_indices = np.random.choice(n_fg_pixels, size = n_fg_clicks, replace = False)
            else:
                # In this case, take all foreground pixels and then obtain some duplicate points by sampling with replacement additionally
                warning_zs[f'z = {slice_index}, foreground'] = n_fg_pixels
                point_indices = np.concatenate([np.arange(n_fg_pixels),
                                            np.random.choice(n_fg_pixels, size = n_fg_clicks-n_fg_pixels, replace = True)])
                
            pos_clicks = slice_fg[point_indices]
            z_col = np.full(n_fg_clicks, slice_index).reshape(n_fg_clicks, 1) # add z column
            pos_clicks = np.hstack([z_col, pos_clicks])

            ## Background points
            struct_element = disk(border_distance)
            slice_dilated = dilation(slice, struct_element)
            border_region = slice_dilated - slice

            slice_border = np.where(border_region)
            slice_border = np.array(slice_border).T

            n_border_pixels = slice_border.shape[0]
            if n_border_pixels >= n_bg_clicks:
                point_indices = np.random.choice(n_border_pixels, size = n_bg_clicks, replace = False)
            else:
                # In this case, take all border pixels and then obtain some duplicate points by additionally sampling with replacement 
                warning_zs[f'z = {slice_index}, border'] = n_border_pixels
                point_indices = np.concatenate([np.arange(n_border_pixels),
                                            np.random.choice(n_border_pixels, size = n_bg_clicks-n_border_pixels, replace = True)])
            neg_clicks = slice_border[point_indices]
            z_col = np.full(n_bg_clicks, slice_index).reshape(n_bg_clicks, 1) # add z column
            neg_clicks = np.hstack([z_col, neg_clicks])

            # Join together the clicks and labels as above.
            clicks, labels = joinShuffleClicks(pos_clicks, neg_clicks)    

            points2D = np.vstack([points2D, clicks])
            labels2D = np.concatenate([labels2D, labels])

        if warning_zs:
            tqdm.write(f'WARNING: some slices in {gt_mask_path} had fewer than n_fg_clicks = {n_fg_clicks} foreground pixels or fewer than n_bg_clicks = {n_bg_clicks} border pixels. Specifically: {warning_zs}')

        prompt_dict[label]['2D'] = (points2D, labels2D)

    # Store prompts for this image
    full_prompt_dict[os.path.basename(gt_mask_path)] = dict(prompt_dict)

# Write results
print(f'Saving prompts to {os.path.join(res_path, "prompts.pkl")}')
os.makedirs(res_path, exist_ok = True)
with open(os.path.join(res_path, 'prompts.pkl'), 'wb') as f:
    pickle.dump(full_prompt_dict, f)
