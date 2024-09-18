from pathlib import Path
import re
import SimpleITK as sitk
from loguru import logger
import numpy as np
from tqdm import tqdm
import json

from intrab.utils.paths import get_dataset_path


dataset_type: dict[str, str] = {
    "Dataset201_MS_Flair_instances": "instance",
    "Dataset209_hanseg_mr_oar": "organ",
    "Dataset501_hntsmrg_pre_primarytumor": "instance",
    "Dataset600_pengwin": "instance",
    "Dataset912_colorectal_livermets": "instance",
    "Dataset921_hcc_tace_lesion": "instance",
    "Dataset913_adrenal_acc_ki67": "instance",
    "Dataset920_hcc_tace_liver": "organ",
    "Dataset911_LNQ_instances": "instance",
    "Dataset930_RIDER_LungCT": "instance",
}


def verify_instance_datasets(ds_dir: Path):
    lbls = ds_dir / "labelsTr"
    for lbl in tqdm(list(lbls.iterdir()), desc=f"Checking {ds_dir.name}"):
        lbl_im = sitk.ReadImage(lbl)
        lbl_arr = sitk.GetArrayFromImage(lbl_im)
        unique_lbls_found = set(np.unique(lbl_arr))
        if set(list(range(0, len(unique_lbls_found)))) != unique_lbls_found:
            logger.warning(f"Labels of {ds_dir.name} not consecutive integers. Got {unique_lbls_found}")
    return


def verify_organ_datasets(ds_dir: Path):
    dataset_json = ds_dir / "dataset.json"
    with open(dataset_json, "r") as f:
        dataset_info = json.load(f)
    labels = dataset_info["labels"]
    label_values = set(labels.values())

    lbls = ds_dir / "labelsTr"
    for lbl in tqdm(list(lbls.iterdir()), desc=f"Checking {ds_dir.name}"):
        lbl_im = sitk.ReadImage(lbl)
        lbl_arr = sitk.GetArrayFromImage(lbl_im)
        unique_lbls_found = set(np.unique(lbl_arr))
        # Check if the labels are in the dataset labels
        if not unique_lbls_found.issubset(label_values):
            logger.warning(
                f"Labels of {ds_dir.name} not in dataset labels. \nGot: {unique_lbls_found}.\nValid labels are: {labels}."
            )
    return


def verify_dataset(ds_dir: Path, ds_type: str):
    """Verify that the dataset has the correct labels for the dataset type."""
    if ds_type == "instance":
        verify_instance_datasets(ds_dir)
    elif ds_type == "organ":
        verify_organ_datasets(ds_dir)
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")


if __name__ == "__main__":
    data_path = get_dataset_path()
    for ds_name, ds_type in dataset_type.items():
        verify_dataset(data_path / ds_name, ds_type)
