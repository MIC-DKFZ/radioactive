import json
import os
from pathlib import Path
from typing import get_args
import yaml
from intrab.model.model_utils import model_registry
from intrab.utils.paths import get_dataset_path
from intrab.prompts.prompter import static_prompt_styles
from loguru import logger




def get_matching_datasets(dataset_path: Path, dataset_id: int) -> list[Path]:
    """Finds all matching datasets by its id and returns a list of paths to that directory."""
    matched_datasets = [dataset_path / d for d in os.listdir(dataset_path) if d.startswith(f"Dataset{dataset_id:03d}")]
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
    assert len(found_matching_datasets) > 0, f"No datasets found in {dataset_path}."
    assert len(found_matching_datasets) == 1, f"Multiple datasets found in {dataset_path}."
    assert found_matching_datasets[0].is_dir(), f"Dataset {dataset_id} is not a directory."
    assert (found_matching_datasets[0] / "dataset.json").exists(), f"Dataset {dataset_id} does not have a dataset.json file."
    return


def sanity_check_dataset_config(dataset_conf: dict) -> None:
    """
    Assert that every configuration has an identifier and a type of either organ or lesion.
    Else raise an AssertionError.
    """
    assert len(dataset_conf) > 0, "No datasets provided in the config file. Please provide at least one dataset."
    for dataset in dataset_conf:
        assert "identifier" in dataset.keys(), f"Dataset identifier is missing: {dataset}."
        assert dataset["type"] in ["organ", "lesion"], f"Dataset {dataset['identifier']} type has to be either organ dataset or lesion dataset."
        verify_dataset_exists(dataset["identifier"])
    return


def sanity_check_model_config(models_conf) -> None:
    """
    Assert that every configuration has an identifier and a type of either organ or lesion.
    Else raise an AssertionError.
    """
    assert len(models_conf) > 0, "No models provided in the config file. Please provide at least one model."
    for model in models_conf:
        assert model in get_args(model_registry), f"Model {model} is not supported. Choose from {get_args(model_registry)}."
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
            assert p in get_args(static_prompt_styles), f"Prompt style {p} is not supported. Choose from {get_args(static_prompt_styles)}." 
    

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
