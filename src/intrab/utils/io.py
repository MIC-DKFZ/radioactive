import json
import os
from pathlib import Path
from typing import get_args
import yaml
from intrab.model.model_utils import model_registry
from intrab.utils.paths import get_dataset_path
from intrab.prompts.prompter import Prompter, static_prompt_styles
from loguru import logger
import nibabel as nib
import numpy as np

from intrab.utils.result_data import PromptResult


def get_matching_datasets(dataset_path: Path, dataset_id: int) -> list[Path]:
    """Finds all matching datasets by its id and returns a list of paths to that directory."""
    matched_datasets = [
        dataset_path / d for d in os.listdir(dataset_path) if d.startswith(f"Dataset{dataset_id:03d}")
    ]
    return matched_datasets


def get_dataset_path_by_id(dataset_id: int) -> Path:
    """Finds the dataset path by its id."""
    dataset_path = get_dataset_path()
    found_matching_datasets = get_matching_datasets(dataset_path, dataset_id)[0]
    return found_matching_datasets


def verify_dataset_exists(dataset_id: int) -> None:
    """
    Verify that the dataset exists.
    """
    dataset_path = get_dataset_path()
    found_matching_datasets = get_matching_datasets(dataset_path, dataset_id)
    assert len(found_matching_datasets) > 0, f"No datasets found in '{dataset_path}'."
    assert len(found_matching_datasets) == 1, f"Multiple datasets found in {dataset_path}."
    assert found_matching_datasets[0].is_dir(), f"Dataset {dataset_id} is not a directory."
    assert (
        found_matching_datasets[0] / "dataset.json"
    ).exists(), f"Dataset {dataset_id} does not have a dataset.json file."
    return


def sanity_check_dataset_config(dataset_conf: dict) -> None:
    """
    Assert that every configuration has an identifier and a type of either organ or lesion.
    Else raise an AssertionError.
    """
    assert len(dataset_conf) > 0, "No datasets provided in the config file. Please provide at least one dataset."
    for dataset in dataset_conf:
        assert "identifier" in dataset.keys(), f"Dataset identifier is missing: {dataset}."
        assert dataset["type"] in [
            "organ",
            "lesion",
        ], f"Dataset {dataset['identifier']} type has to be either organ dataset or lesion dataset."
        verify_dataset_exists(dataset["identifier"])
    return


def sanity_check_model_config(models_conf) -> None:
    """
    Assert that every configuration has an identifier and a type of either organ or lesion.
    Else raise an AssertionError.
    """
    assert len(models_conf) > 0, "No models provided in the config file. Please provide at least one model."
    for model in models_conf:
        assert model in get_args(
            model_registry
        ), f"Model {model} is not supported. Choose from {get_args(model_registry)}."
    return


def sanity_check_prompting(prompt_config: dict) -> None:
    """
    Check if the config is valid.
    """
    assert "type" in prompt_config, "Prompt type is missing."
    assert prompt_config["type"] in ["static", "interactive"], "Prompt type has to be either static or interactive."
    assert len(prompt_config["prompt_styles"]) > 0, "No prompt styles provided."
    if prompt_config["type"] == "static":
        for p in prompt_config["prompt_styles"]:
            assert p in get_args(
                static_prompt_styles
            ), f"Prompt style {p} is not supported. Choose from {get_args(static_prompt_styles)}."


def sanity_check_config(config: dict) -> None:
    """
    Check if the config is valid.
    """
    assert "seeds" in config, f"'Seeds' is missing in config keys. Found: {config.keys()}"
    sanity_check_dataset_config(config["datasets"])
    sanity_check_model_config(config["models"])
    sanity_check_prompting(config["prompting"])


def read_yaml_config(config_path: str) -> dict:
    """
    Read a yaml file and return the dictionary.
    """
    logger.info(f"Reading config from: {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    logger.info(f"Config read successfully.")
    logger.info(f"Sanity checking the config.")
    sanity_check_config(config)
    logger.info(f"Config is valid.")

    return config


def get_img_gts(dataset_dir):
    images_dir = os.path.join(dataset_dir, "imagesTr")
    labels_dir = os.path.join(dataset_dir, "labelsTr")
    imgs_gts = [
        (
            os.path.join(images_dir, img_path),
            os.path.join(labels_dir, img_path.removesuffix("_0000.nii.gz") + ".nii.gz"),
        )
        for img_path in os.listdir(images_dir)  # Adjust the extension as needed
        if os.path.exists(os.path.join(labels_dir, img_path.rstrip("_0000.nii.gz") + ".nii.gz"))
    ]
    return list(sorted(imgs_gts, key=lambda x: x[0]))


def verify_results_dir_exist(targets, results_dir, prompters: list[Prompter]):
    """Creates all output paths if they don't exist"""
    [
        Path(results_dir / p.name / (f"{target_label:03d}" + "__" + target)).mkdir(exist_ok=True, parents=True)
        for p in prompters
        for target, target_label in targets.items()
    ]

    [
        Path(results_dir / "binarised_gts" / (f"{target_label:03d}" + "__" + target)).mkdir(
            exist_ok=True, parents=True
        )
        for target, target_label in targets.items()
    ]


def get_labels_from_dataset_json(dataset_dir: Path) -> dict[str:int]:
    """
    Reads the dataset_json and returns a label dict of {class_name: class_id}
    :param dataset_dir: Path to the dataset directory
    :return: label_dict: A dictionary of {class_name: class_id}
    """
    # Get dataset dict if missing
    with open(dataset_dir / "dataset.json", "r") as f:
        dataset_info = json.load(f)
    label_dict = dataset_info["labels"]
    return label_dict


def binarize_gt(gt_path: Path, label_of_interest: int):
    """
    Creates a binary mask from a multi-class groundtruth in the same spacing.
    """
    gt_nib = nib.load(gt_path)
    gt = gt_nib.get_fdata()
    binary_gt = np.where(gt == label_of_interest, 1, 0)
    binary_gt = nib.Nifti1Image(binary_gt.astype(np.uint8), gt_nib.affine)
    return binary_gt


def create_instance_gt(gt_path: Path) -> tuple[nib.Nifti1Image, nib.Nifti1Image, list[int]]:
    gt_nib: nib.Nifti1Image = nib.load(gt_path)
    gt = gt_nib.get_fdata().astype(np.int16)
    instances = set(np.unique(gt))
    instances = list(instances - {0})
    semantic_gt_np = np.where(gt != 0, 1, 0)
    semantic_gt = nib.Nifti1Image(semantic_gt_np.astype(np.int8), gt_nib.affine)
    instance_gt = gt_nib
    return semantic_gt, instance_gt, instances


def resolve_lesion_overlap(pred_instance: list[PromptResult]) -> list[PromptResult]:
    """
    Removes overlap between the predicted lesion instances.

    DISCLAIMER: Overlapping lesions are not allowed, hence predictions that overlap need to be stratified.
    This is done by taking the lesion with the highest performance, and cutting away that positive region from the worse performing lesion.
    This will artificially inflate the performance of the worse lesion, as false positive regions are removed.
    """
    bin_maps = [pred.predicted_image.get_fdata() for pred in pred_instance]
    bin_map_perf = [pred.perf for pred in pred_instance]
    high_to_low_perf = np.argsort(bin_map_perf)[::-1]
    # ---------- Set previously segmented regions to 0 to avoid overlap ---------- #
    # This has the caveat that if a model considers all lesions as positive instead of the instance, this will make the performance very good.
    disallowed_region = np.zeros_like(bin_maps[0])
    for bin_map_id in high_to_low_perf:
        # Where the disallowed region is 1, set the bin_map to 0
        bin_maps[bin_map_id] = np.where(disallowed_region == 1, 0, bin_maps[bin_map_id])
        disallowed_region = np.logical_or(disallowed_region, bin_maps[bin_map_id])

    for cnt in range(len(pred_instance)):
        pred_instance[cnt].predicted_image = nib.Nifti1Image(bin_maps[cnt], pred_instance[cnt].predicted_image.affine)

    return pred_instance


def save_static_lesion_results(
    prompt_results: list[PromptResult], pred_out: Path, semantic_filename: str, instance_filename: str
):
    """Iterate through all single prompt results that are"""
    semantic_pd_path = pred_out / semantic_filename
    instance_pd_path = pred_out / instance_filename

    prompt_result = resolve_lesion_overlap(prompt_results)

    img = np.zeros_like(prompt_results[0].predicted_image.get_fdata())
    for i, prompt_result in enumerate(prompt_results):
        if i == 0:
            affine = prompt_result.predicted_image.affine
        img = img + np.where(prompt_result.predicted_image.get_fdata() != 0, i + 1, 0)
    semantic_img = nib.Nifti1Image(np.where(img != 0, np.ones_like(img), np.zeros_like(img)), affine, dtype=np.int8)
    instance_img = nib.Nifti1Image(img, affine, dtype=np.int8)

    semantic_pd_path = pred_out / semantic_filename
    instance_pd_path = pred_out / instance_filename
    # ToDo: Save the prompt steps as well.
    instance_img.to_filename(instance_pd_path)
    semantic_img.to_filename(semantic_pd_path)


def save_interactive_lesion_results(
    all_prompt_results: list[list[PromptResult]], pred_out: Path, semantic_filename: str, instance_filename: str
):
    """
    Save the interactive lesion results.
    Each lesion has it's own interactive results. The outer list count goes through the lesions and the inner list goes through the iterations.
    """
    all_arr = []
    affine = None
    for lesion_id, prompt_results in enumerate(all_prompt_results):
        all_arr_iter = []
        prompt_res: PromptResult
        for cnt, prompt_res in enumerate(prompt_results):
            if lesion_id == 0 and cnt == 0:
                affine - prompt_res.predicted_image.affine
            img = (prompt_res.predicted_image.get_fdata()) + lesion_id
            all_arr_iter.append(img)
        all_arr.append(all_arr_iter)
    joint_arr = np.array(all_arr)  # Shape: (num_lesions, num_iterations, x, y, z)
    iter_wise_arr = np.sum(joint_arr, axis=0, keepdims=False)  # Shape: (num_iterations, x, y, z)
    # Save the results iteratively
    for cnt in range(iter_wise_arr.shape[0]):
        semantic_img = nib.Nifti1Image(np.where(np.nonzero(iter_wise_arr[cnt]), 1, 0), affine)
        instance_img = nib.Nifti1Image(iter_wise_arr[cnt], affine)
        semantic_pd_path = pred_out / f"iter_{cnt}" / semantic_filename
        instance_pd_path = pred_out / f"iter_{cnt}" / instance_filename
        semantic_img.to_filename(semantic_pd_path)
        instance_img.to_filename(instance_pd_path)
