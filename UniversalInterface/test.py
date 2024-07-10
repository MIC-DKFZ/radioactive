# Get initial segmentation
import numpy as np
from utils.base_classes import Points

from classes.SAMMed2DClass import SAMMed2DInferer
import utils.analysis as anUt
import utils.prompt as prUt
from utils.image import read_im_gt
from utils.interactivity import iterate_2d

# Obtain model, image, gt
device = 'cuda'
sammed2d_checkpoint_path = "/home/t722s/Desktop/UniversalModels/TrainedModels/sam-med2d_b.pth"
sammed2d_inferer = SAMMed2DInferer(sammed2d_checkpoint_path, device)

img_path = '/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_2img/imagesTr/BDMAP_00000001_0000.nii.gz'
gt_path = '/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_2img/labelsTr/BDMAP_00000001.nii.gz'
img, gt = read_im_gt(img_path, gt_path, 3)

# # Exeriment: Point propagation

# seed = 11121
# n_clicks = 5

# # Get seed prompt and bounds
# seed_point = prUt.get_seed_point(gt, n_clicks, seed)
# slices_to_infer = np.where(np.any(gt, axis=(1,2)))[0]

# segmentation, intial_prompt = prUt.point_propagation(sammed2d_inferer, img, seed_point, slices_to_infer, seed, n_clicks)
# print(anUt.compute_dice(gt,segmentation))

# # Experiment: line interpolation
# n_slices = 5
# interpolation = 'linear'
# point_prompt = prUt.line_interpolation(gt, n_slices, interpolation)

# segmentation = sammed2d_inferer.predict(img, point_prompt)
# anUt.compute_dice(segmentation, gt)


# Iteratively improve from line interpolation
## Generate initial segmentation
n_slices = 5
interpolation = 'linear'
point_prompt = prUt.line_interpolation(gt, n_slices, interpolation)

segmentation = sammed2d_inferer.predict(img, point_prompt)

# Improve
# anUt.compute_dice(segmentation, gt)
# condition = 'dof'
# dof_bound = 90
# segmentation, dof, segmentations, prompts = iterate_2d(sammed2d_inferer, img, gt, segmentation, intial_prompt, 
#                                                                          condition = condition, init_dof = 5, dof_bound = dof_bound, seed = seed, detailed = True, pass_prev_prompts=True)
import torch
import utils.interactivity as interactivity

init_dof = 5
condition = 'dof'
dof_bound = 60
pass_prev_prompts = True
inferer = sammed2d_inferer
initial_prompt = point_prompt
perf_bound = 60


# Generate initial segmentation using seed method

# Variables for contour scribble        
contour_distance = 2
disk_size_range = (0,0)
scribble_length = 0.2

# Rename for clarity
prompt = initial_prompt
dof = init_dof

# Model should not be verbose during this loop. Restore state later
verbosity = inferer.verbose 
inferer.verbose = False

# Obtain low res masks for interactivity

slices_inferred = np.unique(prompt.coords[:,0])
low_res_masks = inferer.slice_lowres_outputs.copy()
low_res_masks = {k:torch.sigmoid(v).squeeze().cpu().numpy() for k,v in low_res_masks.items()}

# Flag for calculating dof
has_generated_positive_prompt = False

# Tracking variables
prompts = [prompt]
segmentations = [segmentation]
dice_scores = [prUt.compute_dice(segmentation, gt)]
max_fp_idxs = []


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
        axis = 1 # Can extend to also check when fixing axis 2
        fp_sums = np.sum(fp_mask, axis=tuple({0,1,2} - {axis}))
        max_fp_idx = np.argmax(fp_sums)
        max_fp_idxs.append(max_fp_idx) # For tracking
        max_fp_slice = gt[:, max_fp_idx]
        slice_seg = segmentation[:, max_fp_idx]

        scribble = interactivity.gen_contour_fp_scribble(max_fp_slice, slice_seg, contour_distance, disk_size_range, scribble_length)
        if scribble is None:
            generate_positive_prompts = 1 # Generate positive prompts instead
        else:  # Otherwise subset scribble to false positives  to generate new prompt
            scribble_coords = np.where(scribble)
            scribble_coords = np.stack(scribble_coords, axis = 1)

            # Obtain false positive points and make new prompt
            is_fp_mask = slice_seg[*scribble_coords.T].astype(bool)
            fp_coords = scribble_coords[is_fp_mask]

            ## Position fp_coords back into original 3d coordinate system
            missing_axis = np.repeat(max_fp_idx, len(fp_coords))
            fp_coords_3d = np.vstack([fp_coords[:,0], missing_axis, fp_coords[:,1]]).T
            improve_slices = np.unique(fp_coords_3d[:,0])
            dof += 3*4 # To dicuss: assume drawing a scribble is as difficult as drawing four points

            if pass_prev_prompts: # new prompt includes old prompts
                ## Add to old prompt
                coords = np.concatenate([prompt.coords, fp_coords_3d], axis = 0)
                labels = np.concatenate([prompt.labels, [0]*len(fp_coords_3d)])
                prompt = Points(coords = coords, labels = labels)

                ## Subset to prompts only on the slices with new prompts
                fix_slice_mask = np.isin(prompt.coords[:,0], improve_slices)
                new_prompt = Points(coords = coords[fix_slice_mask], labels = labels[fix_slice_mask])
            else:
                new_prompt = Points(coords = fp_coords_3d, labels = [0]*len(fp_coords_3d))

    if generate_positive_prompts:
        if not has_generated_positive_prompt: 
            dof += 6 # If first time generating positive prompts, generate a bottom and top point, taking 4 degrees of freedom: (4 dof even though there are 6 coordinates since the coordinate of the lowest and highest slice is fixed) 
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
        dof += 3

        # Interpolate linearly from botom_seed-prompt to top_seed_prompt through the new middle prompt to get new positive prompts
        new_seed_prompt = np.vstack([bottom_seed_prompt, new_middle_seed_prompt, top_seed_prompt])
        new_coords =  prUt.interpolate_points(new_seed_prompt, kind = 'linear').astype(int)

        if pass_prev_prompts:
            # Add to old prompt
            coords = np.concatenate([prompt.coords, new_coords], axis = 0)
            labels = np.concatenate([prompt.labels, [1]*len(new_coords)])
            prompt = Points(coords = coords, labels = labels)
            new_prompt = prompt
        else:
            prompt = Points(coords = new_coords, labels = [1]*len(new_coords))
            new_prompt = prompt
        
        improve_slices = slices_inferred # improve all slices 

    # Generate new segmentation and integrate into old one
    new_seg = inferer.predict(img, new_prompt, low_res_masks)
    prompts.append(new_prompt)
    segmentation[improve_slices] = new_seg[improve_slices]
    segmentations.append(segmentation.copy())

    # Update the trackers
    low_res_masks.update({fix_slice_idx: torch.sigmoid(inferer.slice_lowres_outputs[fix_slice_idx]).squeeze().cpu().numpy() for fix_slice_idx in improve_slices})
    dice_scores.append(prUt.compute_dice(segmentation, gt))
    print(dice_scores[-1])
    

    # Check break conditions
    if condition == 'dof' and dof >= dof_bound:
        if inferer.verbose:
            print(f'degrees of freedom bound met; terminating with performance {dice_scores[-1]}')
        break
    elif condition == 'perf' and dice_scores[-1] >= perf_bound:
        if inferer.verbose:
            print(f'performance bound met; terminating with performance {dice_scores[-1]}')
        break
    elif len(dice_scores) == 10:
        if inferer.verbose:
            print(f'Could not achieve desired performance/dof within 10 steps; terminating with performance {dice_scores[-1]}')
        break
inferer.verbose = verbosity
