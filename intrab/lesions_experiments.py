# Experiments content
import os
import numpy as np
import json

from intrab.utils.analysis import compute_dice
from intrab.prompts.prompt_3d import get_bbox3d, get_pos_clicks3D
from intrab.prompts.prompt_utils import get_bbox3d_sliced, get_pos_clicks2D_row_major

from tqdm import tqdm
import shutil
import nibabel as nib
import logging


def get_experiments(inferer, seed):
    experiments = {}
    # Add point prompting methods depending on dimension
    if "point" in inferer.supported_prompts:
        if inferer.dim == 2:
            experiments.update(
                {"random_points": lambda organ_mask: get_pos_clicks2D_row_major(organ_mask, 1, seed=seed)}
            )

        if inferer.dim == 3:
            experiments.update({"random_points_3d": lambda organ_mask: get_pos_clicks3D(organ_mask, 1, seed)})

    # Add box prompting methods depending on dimension
    if "box" in inferer.supported_prompts:
        if inferer.dim == 2:
            experiments.update({"bbox3d_sliced": lambda organ_mask: get_bbox3d_sliced(organ_mask)})
        if inferer.dim == 3:
            experiments.update({"bbox3d": lambda organ_mask: get_bbox3d(organ_mask)})
    return experiments


def run_experiments(inferer, imgs_gts, results_dir, save_segs=False, seed=11121):
    inferer.verbose = False  # No need for progress bars per inference
    results_dir = results_dir.removesuffix("/")

    # Define experiments
    experiments = get_experiments(inferer, seed)
    exp_names = list(experiments.keys())

    # Initialize results dictionary, create results folders and configure logging
    loggers = {}
    for exp_name in exp_names:
        exp_results_dir = results_dir + "_" + exp_name
        if os.path.exists(exp_results_dir):
            shutil.rmtree(exp_results_dir)
        os.makedirs(exp_results_dir, exist_ok=True)

        # Set up logger - needs extra handling to permit multiple logs
        loggers[exp_name] = logging.getLogger(exp_name)
        log_file_path = os.path.join(exp_results_dir, "generate_segmentations.log")
        handler = logging.FileHandler(log_file_path, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        loggers[exp_name].addHandler(handler)
        loggers[exp_name].setLevel(logging.INFO)

    results = {
        exp_name: {  # For now, create one dictionary. This will be split by experiment later
            split: {os.path.basename(gt): {} for img, gt in imgs_gts_split}
            for split, imgs_gts_split in imgs_gts.items()
        }
        for exp_name in exp_names
    }

    # Configure logging

    # Generate segmentations
    overall_status = "All segmentations successfully generated"

    # Loop through all image and label pairs
    for split in ["Tr", "Ts"]:
        for img_path, gt_path in tqdm(imgs_gts[split], desc="looping through files\n", leave=False):
            base_name = os.path.basename(gt_path)

            inferer.set_image(img_path)
            gt = nib.load(gt_path).get_fdata()

            instances_present = np.unique(gt).astype(int)
            instances_present = instances_present[instances_present != 0]  # remove background

            # Loop through each instance of lesion present
            for instance in tqdm(instances_present, leave=False, desc="looping through instances"):
                organ_mask = np.where(gt == instance, 1, 0)

                # Handle experiment. Kept in a loop despite being one item for simplicity of adapting code from original
                for exp_name, prompting_func in experiments.items():
                    prompt = prompting_func(organ_mask)
                    try:
                        segmentation = inferer.predict(prompt)
                        dice_score = compute_dice(segmentation.get_fdata(), organ_mask)

                        if save_segs:
                            seg_dir = os.path.join(
                                results_dir + "_" + exp_name,
                                exp_name,
                                "segmentations" + split,
                                base_name.removesuffix(".nii.gz"),
                            )
                            os.makedirs(seg_dir, exist_ok=True)
                            segmentation.to_filename(
                                os.path.join(seg_dir, "instance_" + str(instance) + "_seg.nii.gz")
                            )

                    except Exception as e:
                        dice_score = None
                        overall_status = "Some segmentations failed"
                        loggers[exp_name].error(
                            f"Segmenting instance {instance} in image {base_name} failed: {str(e)}"
                        )
                    results[exp_name][split][base_name][str(instance)] = dice_score

    # Save results
    for exp_name in exp_names:
        results_path = os.path.join(results_dir + "_" + exp_name, "results.json")
        with open(results_path, "w") as f:
            json.dump(results[exp_name], f, indent=4)

    logging.info(overall_status)
    print(
        f"Segmentations generated and saved to {[results_dir + '_' + exp_name for exp_name in exp_names]}; {overall_status}"
    )

    return exp_names


def run_postprocess(results_dir, exp_names, dataset_dir):
    results_dir = results_dir.removesuffix("/")
    status = "All segmentations could be merged"

    for exp_name in exp_names:
        exp_results_dir = results_dir + "_" + exp_name
        with open(os.path.join(exp_results_dir, "results.json"), "r") as f:
            results = json.load(f)
        results["summary_results"] = {}
        # obtain results json that we will add dice scores to
        dice_all_list = []
        for split in ["Tr", "Ts"]:
            dice_split_list = []
            # Obtain folders of segmentations
            seg_dir_parent = os.path.join(exp_results_dir, exp_name, "segmentations" + split)
            if not os.path.exists(seg_dir_parent):
                continue

            seg_dirs = [
                os.path.join(seg_dir_parent, f) for f in os.listdir(seg_dir_parent)
            ]  # Obtain all segmentation folders (ie each one containing all the instance segmentations for a given image
            seg_dirs = [
                d for d in seg_dirs if not d.endswith("merged")
            ]  # Ignore merged folders from possible previous attempts at postprocessing
            os.makedirs(
                os.path.join(seg_dir_parent, "merged"), exist_ok=True
            )  # make folder to store merged niftis in

            for seg_dir in seg_dirs:
                num_instances_segmented = len(os.listdir(seg_dir))
                if num_instances_segmented == 0:
                    continue  # Skip if there are no segmentations (ie no foreground)

                gt_basename = os.path.basename(seg_dir) + ".nii.gz"

                try:
                    gt_path = os.path.join(dataset_dir, "labels" + split, gt_basename)
                    gt = nib.load(gt_path).get_fdata().astype(np.uint8)
                    if not num_instances_segmented == len(np.unique(gt)) - 1:  # Subtract one to eliminate foreground
                        merged_dice = None
                        raise RuntimeError("Could not merge instances, missing segmentations.")
                    gt = np.where(gt > 0, 1, 0)

                    # Obtain merged image
                    segs = [os.path.join(seg_dir, f) for f in os.listdir(seg_dir)]

                    summed_image = None

                    for seg_path in segs:
                        # Load the NIfTI file using nibabel
                        img = nib.load(seg_path)
                        img_data = img.get_fdata()

                        if summed_image is None:  # Initialize the summed_image with the first image data
                            summed_image = img_data.copy()
                        else:  # Add the current image data to the summed_image
                            if img_data.shape != summed_image.shape:
                                raise ValueError("All instance segmentations should have the same dimensions")
                            summed_image += img_data

                    merged_image = np.where(summed_image > 0, 1, 0).astype(np.uint8)

                    ## Save image
                    merged_nifti = nib.Nifti1Image(merged_image, affine=img.affine, header=img.header)
                    merged_nifti.to_filename(os.path.join(seg_dir_parent, "merged", gt_basename))

                    # Obtain new dice scores
                    merged_dice = compute_dice(gt, merged_image)
                    dice_split_list.append(merged_dice)
                except:
                    merged_dice = None
                    status = "Some segmentations could not be merged"

                results[split][gt_basename]["all"] = merged_dice

            results["summary_results"][split] = np.mean(dice_split_list)
            dice_all_list.extend(dice_split_list)

        results["summary_results"]["all_splits"] = np.mean(dice_all_list)

        with open(os.path.join(exp_results_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

    print(f"Postprocessing finished: {status}")
