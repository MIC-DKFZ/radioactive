# Get initial segmentation
from classes.SAMMed2DClass_unstable import SAMMed2DInferer
from utils.base_classes import Points
import torch

import utils.analysis as anUt
import utils.prompt as prUt
from utils.image import read_im_gt

import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils.interactivity import gen_contour_fp_scribble

# Obtain model, image, gt
device = 'cuda'
sammed2d_checkpoint_path = "/home/t722s/Desktop/UniversalModels/TrainedModels/sam-med2d_b.pth"
sammed2d_inferer = SAMMed2DInferer(sammed2d_checkpoint_path, device)

img_path = '/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_2img/imagesTr/BDMAP_00000001_0000.nii.gz'
gt_path = '/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_2img/labelsTr/BDMAP_00000001.nii.gz'
img, gt = read_im_gt(img_path, gt_path, 3)

# Obtain initial segmentation
# Experiment: Point propagation

seed = 11121
n_clicks = 5

# Get seed prompt and bounds
seed_point = prUt.get_seed_point(gt, n_clicks, seed)
slices_to_infer = np.where(np.any(gt, axis=(1,2)))[0]

segmentation, all_prompts = prUt.point_propagation(sammed2d_inferer, img, seed_point, slices_to_infer, seed, n_clicks)
print(anUt.compute_dice(gt,segmentation))

anUt.compute_dice(segmentation, gt)


with open('/home/t722s/Desktop/test/test_segjhu10.pkl', 'wb') as f:
    pickle.dump(segmentation, f)

with open('/home/t722s/Desktop/test/test_segjhu10.pkl', 'rb') as f:
    segmentation = pickle.load(f)
    
condition = 'dof'
dof_bound = 60
perf_bound = 1 # Place holder, only needed when condition = 'perf'
init_dof = 5
contour_distance = 2
disk_size_range = (0,0)
scribble_length = 0.3


# Initialise low res masks to provide for interactivity
verbosity = sammed2d_inferer.verbose 
sammed2d_inferer.verbose = False
dof = init_dof
slices_inferred = np.unique(all_prompts.value['coords'][:,0])
low_res_masks = sammed2d_inferer.slice_lowres_dict.copy()
low_res_masks = {k:torch.sigmoid(v).squeeze().cpu().numpy() for k,v in low_res_masks.items()}

has_generated_positive_prompt = False

prompts = [all_prompts]
segmentations = [segmentation]
dice_scores = [prUt.compute_dice(segmentation, gt)]


while True:
    # Determine whether to give positive prompts or attempt negative prompt
    fn_mask = (segmentation == 0) & (gt == 1)
    fn_count = np.sum(fn_mask)

    fg_count = np.sum(segmentation)

    generate_positive_prompts_prob = fn_count/fg_count # Generate positive prompts when much of the foreground isn't segmented
    generate_positive_prompts = np.random.binomial(1,generate_positive_prompts_prob)

    if not generate_positive_prompts:
        # Obtain contour scribble on worst sagittal slice
        fp_mask = (segmentation == 1) & (gt == 0)
        axis = 1
        fp_sums = np.sum(fp_mask, axis=tuple({0,1,2} - {axis}))
        max_fp_idx = np.argmax(fp_sums)
        max_fp_slice = gt[:, max_fp_idx]
        slice_seg = segmentation[:, max_fp_idx]

        scribble = gen_contour_fp_scribble(gt[:, max_fp_idx], slice_seg, contour_distance, disk_size_range, scribble_length)
        if scribble is None:
            generate_positive_prompts = 1 # Generate positive prompts instead
        else: 
            scribble_coords = np.where(scribble)
            scribble_coords = np.stack(scribble_coords, axis = 1)

            # Obtain false positive points and make new prompt
            is_fp_mask = slice_seg[*scribble_coords.T].astype(bool)
            fp_coords = scribble_coords[is_fp_mask]

            ## Position fp_coords back into original 3d coordinate system
            missing_axis = np.repeat(max_fp_idx, len(fp_coords))
            fp_coords_3d = np.vstack([fp_coords[:,0], missing_axis, fp_coords[:,1]]).T
            improve_slices = fp_coords_3d[:,0]
            dof += 3*4 # To dicuss: assume drawing a scribble is as difficult as drawing four points

            ## Add to old prompt
            coords = np.concatenate([all_prompts.value['coords'], fp_coords_3d], axis = 0)
            labels = np.concatenate([all_prompts.value['labels'], [0]*len(fp_coords_3d)])
            all_prompts = Points(value = {'coords': coords, 'labels': labels})

            ## Subset to prompts only on the slices with new prompts
            coords, labels = all_prompts.value.values()
            fix_slice_mask = np.isin(all_prompts.value['coords'][:,0], improve_slices)
            new_prompt = Points({'coords': coords[fix_slice_mask], 'labels': labels[fix_slice_mask]})

    if generate_positive_prompts:
        if not has_generated_positive_prompt:
            dof+=6 
            bottom_seed_prompt, _, top_seed_prompt = prUt.get_fg_points_from_cc_centers(gt, 3)
            has_generated_positive_prompt = True
        # Find fp coord from the middle axial range of the image
        lower, upper = np.percentile(slices_inferred, [30, 70 ])
        fp_coords = np.vstack(np.where(fn_mask)).T
        middle_mask = (lower < fp_coords[:, 0]) & (fp_coords[:,0] < upper) # Mask to determine which false negatives lie between the 30th to 70th percentile
        if np.sum(middle_mask) == 0:
            middle_mask = np.ones(len(fp_coords), bool) # If there are no false negatives in the middle, draw from all coordinates (unlikely given that there must be many)
        fp_coords = fp_coords[middle_mask, :]
        new_middle_seed_prompt = fp_coords[np.random.choice(len(fp_coords), 1)]

        # Obtain top and bottom prompts and then interpolate a line of coordinates in between
        
        dof += 3

        new_seed_prompt = np.vstack([bottom_seed_prompt, new_middle_seed_prompt, top_seed_prompt])
        new_coords =  prUt.interpolate_points(new_seed_prompt, kind = 'linear').astype(int)

        # Add to old prompt
        coords = np.concatenate([all_prompts.value['coords'], new_coords], axis = 0)
        labels = np.concatenate([all_prompts.value['labels'], [1]*len(new_coords)])
        all_prompts = Points(value = {'coords': coords, 'labels': labels})
        new_prompt = all_prompts # When generating positive prompts, prompts are generated for all slices
        improve_slices = slices_inferred # improve all slices

    # Generate new segmentation and integrate into old one
    new_seg = sammed2d_inferer.predict(img, new_prompt)
    prompts.append(new_prompt)
    segmentation[improve_slices] = new_seg[improve_slices]
    segmentations.append(segmentation.copy())
    # Update the dictionary
    low_res_masks.update({fix_slice_idx: torch.sigmoid(sammed2d_inferer.slice_lowres_dict[fix_slice_idx]).squeeze().cpu().numpy() for fix_slice_idx in improve_slices})
    dice_scores = [prUt.compute_dice(segmentation, gt)]
    print(dice_scores[-1])

    # Check break conditions
    if condition == 'dof' and dof >= dof_bound:
        print(f'degrees of freedom bound met; terminating with performance {dice_scores[-1]}')
        break
    elif condition == 'perf' and dice_scores[-1] >= perf_bound:
        print(f'performance bound met; terminating with performance {dice_scores[-1]}')
        break
    elif condition == 'perf' and len(dice_scores) == 10:
        print(f'Could not achieve desired performance within 10 steps; terminating with performance {dice_scores[-1]}')

sammed2d_inferer.verbose = verbosity # return verbosity to initial state