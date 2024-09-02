import os
import nibabel as nib
import numpy as np

import json
import argparse

from intrab.utils.analysis import compute_dice


# ToDo: What does this snippet do? 
#   Needs to be integrated into the normal workflow and not be just a standalone script.
#   At least split into more functional parts.
if __name__ == '__main__':
    # Merge segmentations into one binary mask and obtain and save the merged dice
    # results_dir = '/home/t722s/Desktop/ExperimentResults_lesions/sam_melanoma_HD_20240729_1724/'
    # dataset_dir = '/home/t722s/Desktop/Datasets/melanoma_HD'

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type = str, required = True, help = 'Select from "sam", "medsam", "sammed2d"')
    parser.add_argument('-d', '--dataset_dir', type = str, required = True, help = 'Path to dataset')
    parser.add_argument('-r', '--results_dir', type = str, required = True, help = 'Path to desired results directory')
    args = parser.parse_args()

    model_name = args.model_name
    dataset_dir = args.dataset_dir
    results_dir = args.results_dir

    dataset_name = os.path.basename(dataset_dir.removesuffix('/'))
    results_dir = os.path.join(results_dir, model_name + '_' + dataset_name)

    seg_dirs = []
    for split in ['Tr', 'Ts']:
        seg_dir_parent = os.path.join(results_dir, 'segmentations' + split)
        if os.path.exists(seg_dir_parent):
            seg_dirs.extend([os.path.join(seg_dir_parent, f) for f in os.listdir(seg_dir_parent)])
    seg_dirs = [d for d in seg_dirs if os.path.isdir(d)] # Subset to folders


    # obtain results json that we will add dice scores to
    with open(os.path.join(results_dir, 'results.json'), 'r') as f:
        results = json.load(f)

    results['summary_results'] = {}
    
    total_dice_all = []
    for split in ['Tr', 'Ts']:
        total_dice_split = []
        # Obtain folders of segmentations
        seg_dir_parent = os.path.join(results_dir, 'segmentations' + split)
        if not os.path.exists(seg_dir_parent):
            continue

        os.makedirs(os.path.join(seg_dir_parent, 'merged'), exist_ok=True) # make folder to store merged niftis in.
        seg_dirs = [os.path.join(seg_dir_parent, f) for f in os.listdir(seg_dir_parent)]

        for seg_dir in seg_dirs:
            if len(os.listdir(seg_dir)) == 0:
                continue # Skip if there are no segmentations (ie no foreground)

            gt_basename = os.path.basename(seg_dir) + '.nii.gz'

            try: 
                # Obtain merged image
                segs = [os.path.join(seg_dir, f) for f in os.listdir(seg_dir)]
                
                summed_image = None

                for seg_path in segs:
                    # Load the NIfTI file using nibabel
                    img = nib.load(seg_path)
                    img_data = img.get_fdata()
                    
                    if summed_image is None:
                        # Initialize the summed_image with the first image data
                        summed_image = img_data.copy()
                    else:
                        # Check if the current image has the same shape as the summed_image
                        if img_data.shape != summed_image.shape:
                            raise ValueError("All images must have the same dimensions")
                        # Add the current image data to the summed_image
                        summed_image += img_data

                merged_image = np.where(summed_image>0, 1, 0).astype(np.uint8)

                ## Save image
                merged_nifti = nib.Nifti1Image(merged_image, affine=img.affine, header=img.header)
                merged_nifti.to_filename(os.path.join(seg_dir_parent, 'merged', gt_basename))

                # Obtain new dice scores
                ## Obtain and binarize gt
                gt_path = os.path.join(dataset_dir, 'labels' + split, gt_basename)
                gt = nib.load(gt_path).get_fdata().astype(np.uint8)
                gt = np.where(gt > 0, 1, 0)

                total_dice = compute_dice(gt, merged_image)
                total_dice_split.append(total_dice)
            except:
                total_dice = None

            results[split][gt_basename]['all'] = total_dice

        results['summary_results'][split] = np.mean(total_dice_split)
        total_dice_all.extend(total_dice_split)

    results['summary_results']['all_splits'] = np.mean(total_dice_all)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent = 4)
        



        

