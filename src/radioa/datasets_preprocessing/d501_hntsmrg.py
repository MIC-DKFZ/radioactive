import json
import os
from pathlib import Path
import shutil
import numpy as np
from toinstance import InstanceNrrd
import glob

from tqdm import tqdm

from radioa.datasets_preprocessing.utils import (
    copy_images,
    copy_labels_of_modality_and_transform_to_instance,
    download_from_zenodo,
)
from radioa.utils.io import get_dataset_path_by_id
from radioa.utils.paths import get_dataset_path
import SimpleITK as sitk


def preprocess(raw_dataset_path: Path):

    # This dataset contains two semantic:
    # - Primary Tumor (Class 1)
    # - Metastatic Lymph Nodes (Class 2)
    # We will create instances of the first semantic class through connected components.
    #   Adding the class of metastatic lymph nodes can be done later if necessary.

    # IMPORTANT: In the dataset description they mention at MOST one primary tumor per patient. (Class 1)
    #   This means no Connected Component analysis has to be done for Class 1
    # ! IF THIS GETS EXTENDED TO CLASS 2 A CC NEEDS TO BE ADDED !

    # Did some mess up with the folder structure, but whatever.
    data_path = raw_dataset_path / "HNTSMRG24_train" / "HNTSMRG24_train"
    target_path = get_dataset_path() / "Dataset501_hntsmrg_pre_primarytumor"
    (target_path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (target_path / "labelsTr").mkdir(parents=True, exist_ok=True)

    img_paths = list(data_path.rglob("preRT/*preRT_T2.nii.gz"))
    lbl_paths = list(data_path.rglob("preRT/*preRT_mask.nii.gz"))

    # SOME IMAGES DON'T HAVE LABELS, BUT SINCE WE DO INSTANCE STUFF WE IGNORE THEM
    #   This is currently not optimal but we don't support empty cases.

    images = {}
    for img in tqdm(img_paths, desc="Indexing images"):
        case_id = Path(img).name.replace(".nii.gz", "").replace("_T2", "")
        target_img_name = (
            target_path / "imagesTr" / (Path(img).name.replace(".nii.gz", "_0000.nii.gz").replace("_T2", ""))
        )
        images[case_id] = {"target_path": target_img_name, "img_path": img}

    labels = {}
    for lbl in tqdm(lbl_paths, desc="Indexing labels"):
        case_id = Path(lbl).name.replace("_mask", "").replace(".nii.gz", "")
        target_lbl_name = target_path / "labelsTr" / (case_id + ".nii.gz")

        img = sitk.ReadImage(lbl)
        img_array = sitk.GetArrayFromImage(img)
        # Remove the second label
        img_array[img_array == 2] = 0
        if np.any(img_array):
            labels[case_id] = {"target_path": target_lbl_name, "lbl_path": lbl}

    for case_id, label in tqdm(labels.items(), desc="Creating instances of non-zero images"):
        # Read the label, remove the second label and save it
        img = sitk.ReadImage(label["lbl_path"])
        img_array = sitk.GetArrayFromImage(img)
        img_array = img_array.astype(np.uint8)
        img_array[img_array == 2] = 0  # Only has lbl 1 and 2, remove 2 to only have 1 left
        new_img = sitk.GetImageFromArray(img_array)
        new_img.CopyInformation(img)
        sitk.WriteImage(new_img, label["target_path"])
        # Get the associated image and save that too.
        img = shutil.copy(images[case_id]["img_path"], images[case_id]["target_path"])

    # ------------------------------- Dataset Json ------------------------------- #
    with open(get_dataset_path() / "Dataset501_hntsmrg_pre_primarytumor" / "dataset.json", "w") as f:
        json.dump(
            {
                "channel_names": {"0": "T2 MRI"},
                "labels": {"background": 0, "Primary Tumor": 1},
                "numTraining": len(labels),
                "file_ending": ".nii.gz",
                "name": "HNTS-MRG 24 Challenge - Primary Tumor instances",
                "reference": "https://hntsmrg24.grand-challenge.org/dataset/",
                "release": "https://zenodo.org/records/11199559",
            },
            f,
        )


def main():
    """
    The BrainMetShare dataset is a collection of 3D MRI scans of brain metastases.
    Originally the dataset comes with four modalities, but the current models only take one.

    We create two versions of this dataset:
    - One using T1 gradient echo post-contrast modality.
    - One using T1 spin echo post-contrast modality.
    """

    # Need modality 0 (gradient echo post ce ) and 2 (spin echo post ce)
    data_path = get_dataset_path_by_id(501)

    target_path_pt = data_path.parent / "Dataset502_hntsmrg_pre_primarytumor"
    target_path_mln = data_path.parent / "Dataset503_hntsmrg_pre_metastatic_lymphnodes"

    copy_images(data_path, target_path_pt)
    copy_labels_of_modality_and_transform_to_instance(
        1,
        data_path,
        target_path_pt,
        dataset_json_description="HNTSMRG dataset's Primary Tumor instances dataset.",
    )
    copy_images(data_path, target_path_mln)
    copy_labels_of_modality_and_transform_to_instance(
        2,
        data_path,
        target_path_mln,
        dataset_json_description="HNTSMRG dataset's Metastatic Lymph Nodes instances dataset.",
    )


if __name__ == "__main__":
    main()
