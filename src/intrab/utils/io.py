import json
import os
from pathlib import Path
from typing import Any, get_args
import yaml
from intrab.model.model_utils import model_registry
from intrab.utils.paths import get_dataset_path
from intrab.prompts.prompter import Prompter, static_prompt_styles
from loguru import logger
import nibabel as nib
import SimpleITK as sitk
from nrrd import NRRDHeader, read, write
import numpy as np
import tempfile
from toinstance import InstanceNrrd

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


def get_img_gts(dataset_dir: str) -> list[tuple[str,str]]:
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


def verify_results_dir_exist(targets, results_dir, prompters: list[Prompter]) -> None:
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


def binarize_gt(gt_path: Path, label_of_interest: int) -> nib.Nifti1Image:
    """
    Creates a binary mask from a multi-class groundtruth in the same spacing.
    """
    gt_nib = nib.load(gt_path)
    gt = gt_nib.get_fdata()
    binary_gt = np.where(gt == label_of_interest, 1, 0)
    binary_gt = nib.Nifti1Image(binary_gt.astype(np.uint8), gt_nib.affine)
    return binary_gt


def create_instance_gt(gt_path: Path) -> tuple[nib.Nifti1Image, list[int]]:
    gt_nib: nib.Nifti1Image = nib.load(gt_path)
    gt = gt_nib.get_fdata().astype(np.int16)
    instances = set(np.unique(gt))
    instances = list(instances - {0})
    instance_gt = gt_nib
    return instance_gt, instances


def nifti_to_nrrd(nifti: nib.Nifti1Image) -> tuple[np.ndarray, dict]:
    """
    Convert a nifti image to nrrd.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # What a mess
        nifti.to_filename(tmpdir + "/tmp.nii.gz")
        instance_sitk = sitk.ReadImage(tmpdir + "/tmp.nii.gz")
        sitk.WriteImage(instance_sitk, tmpdir + "/tmp.nrrd")
        nrrd_arr, nrrd_header = read(tmpdir + "/tmp.nrrd")
    return nrrd_arr, nrrd_header


def create_innrrd_instance_im(prompt_results: list[PromptResult]) -> InstanceNrrd:
    """
    Create a nrrd instance groundtruth from the prompt results.
    """

    # Need header information.
    _, nrrd_header = nifti_to_nrrd(prompt_results[0].predicted_image)
    prompt_result_bin_maps = [prompt_result.predicted_image.get_fdata() for prompt_result in prompt_results]

    innrrd = InstanceNrrd.from_binary_instance_maps({1: prompt_result_bin_maps}, nrrd_header)

    return innrrd


def save_nib_instance_gt_as_nrrd(nib_gt: nib.Nifti1Image, out_path: Path):
    """
    Save the instance groundtruth as a nrrd file.
    """
    nrrd_arr, nrrd_header = nifti_to_nrrd(nib_gt)
    innrrd = InstanceNrrd.from_instance_map(nrrd_arr, nrrd_header, class_name=1)
    innrrd.to_file(out_path)


def save_static_lesion_results(
    prompt_results: list[PromptResult] | list[list[PromptResult]],
    pred_out: Path,
    instance_filename: str,
):
    """Iterate through all single prompt results that are"""
    innrrd_im = create_innrrd_instance_im(prompt_results)
    pred_out.mkdir(exist_ok=True, parents=True)
    instance_pd_path = pred_out / instance_filename
    innrrd_im.to_file(instance_pd_path)


def save_interactive_lesion_results(
    all_prompt_results: list[list[PromptResult]], pred_out: Path, instance_filename: str
):
    """
    Save the interactive lesion results.
    Each lesion has it's own interactive results. The outer list count goes through the lesions and the inner list goes through the iterations.
    """

    # Outer list holds the lesions, inner list holds the iterations.
    n_iterations = len(all_prompt_results[0])

    for n_iters in range(n_iterations):
        same_iter_prompt_results = [apr[n_iters] for apr in all_prompt_results]
        save_static_lesion_results(same_iter_prompt_results, pred_out / f"iter_{n_iters}", instance_filename)
