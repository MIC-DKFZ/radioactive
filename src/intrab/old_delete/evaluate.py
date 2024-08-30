from argparse import ArgumentParser
import os
from pathlib import Path
import shutil
from loguru import logger
import nibabel as nib
import numpy as np
from intrab.utils.io import get_dataset_path_by_id, get_img_gts, get_labels_from_dataset_json, read_yaml_config
from intrab.utils.paths import get_results_path
from nneval.evaluate_semantic import semantic_evaluation
from intrab.model.model_utils import inferer_registry, checkpoint_registry, model_registry

def get_imgs_gts(dataset_dir):
    imgs_gts = {"Tr": [], "Ts": []}
    for suffix in ["Tr", "Ts"]:
        images_dir = os.path.join(dataset_dir, "images" + suffix)
        labels_dir = os.path.join(dataset_dir, "labels" + suffix)
        if os.path.exists(images_dir):
            imgs_gts[suffix].extend(
                [
                    (
                        os.path.join(images_dir, img_path),
                        os.path.join(labels_dir, img_path.removesuffix("_0000.nii.gz") + ".nii.gz"),
                    )
                    for img_path in os.listdir(images_dir)  # Adjust the extension as needed
                    # if os.path.exists(os.path.join(labels_dir, img_path.removesuffix('_0000.nii.gz') + '.nii.gz')) # Remove check. All the files should exist.
                ]
            )

    return imgs_gts

def write_binarised_gts(dataset_dir: Path, output_dir: Path, targets_dict: dict[str, int]) -> None:
    """
    Given a dataset directory and an output_dir, binarises ground truths based upon a dictionary of targets of relevance, and sorts and stores them in a created subdirectory of output_dir
    """
    # Obtain imgs_gts; only gts are of interest
    imgs_gts = get_imgs_gts(dataset_dir)

    # Main loop
    for split in ['Tr', 'Ts']:
        for _, gt_path in imgs_gts[split]:
            gt_nib = nib.load(gt_path)
            gt = gt_nib.get_fdata()

            for target, label in targets_dict.items():
                # Create directory for gts binarised to this target
                target_dir = output_dir / target
                target_dir.mkdir(exist_ok=True)

                # Binarise and save gt to appropriate folder
                gt_binarised = np.where(gt == label, 1, 0)
                gt_binarised = nib.Nifti1Image(gt_binarised.astype(np.float32), gt_nib.affine)
                gt_binarised.to_filename(target_dir / os.path.basename(gt_path))

    return 

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to a configuration '.yaml' file to run the experiments with.",
    )
    args = parser.parse_args()
    
    config = read_yaml_config(args.config)

    for seed in config["seeds"]: # Currently would overwrite previous runs, I believe. Insert into results path?
        model_name: model_registry
        for dataset in config["datasets"]:
            logger.info(f"Loading dataset {dataset['identifier']:03d}")
            dataset_root: Path = get_dataset_path_by_id(dataset["identifier"])
            logger.info(f"Dataset loaded from {dataset_root}")
            dataset_name: str = dataset_root.name

            # ToDo: Add the excluded class ids here.
            label_dict: dict[str, int] = get_labels_from_dataset_json(dataset_root)
            label_dict = {k:v for k,v in label_dict.items() if k!='background'} # Hardcode until excluding classes is implemented
            imgs_gts: list[tuple[str, str]] = get_img_gts(dataset_root)

            # Write binarised gts to disk, if not already done
            binarised_gts_parent_dir = get_results_path() 
            if not binarised_gts_parent_dir.exists():
                binarised_gts_parent_dir.mkdir() / '_binarised_gts'
                write_binarised_gts(dataset_root, binarised_gts_parent_dir, label_dict)

            # Calculate metrics per model, experiment, and target 
            for model_name in config["models"]:              
                for prompter_name in config["prompting"]["prompt_styles"]:
                    for target in label_dict.keys():
                        exp_target_dir = Path(get_results_path() / (model_name + "_" + dataset_name) / prompter_name / target)

                        semantic_evaluation(semantic_pd_path = exp_target_dir, semantic_gt_path = binarised_gts_parent_dir,
                                            output_path = exp_target_dir, classes_of_interest = label_dict.values())
                        
            # shutil.rmtree(binarised_gts_parent_dir) # For now, can keep in case of bugs, but intend to remove later
                    
                    
