import json
import os
from pathlib import Path
import shutil
from toinstance import InstanceNrrd
import glob

from tqdm import tqdm

from intrab.datasets_preprocessing.utils import (
    copy_images,
    copy_labels_of_modality_and_transform_to_instance,
    download_from_zenodo,
)
from intrab.utils.io import get_dataset_path_by_id
from intrab.utils.paths import get_dataset_path
import SimpleITK as sitk
from tempfile import TemporaryDirectory


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

    for img in tqdm(img_paths, desc="Copying images"):
        target_img_name = (
            target_path / "imagesTr" / (Path(img).name.replace(".nii.gz", "_0000.nii.gz").replace("_T2", ""))
        )
        if not target_img_name.exists():
            shutil.copy(img, target_img_name)

    for lbl in tqdm(lbl_paths, desc="Copying labels"):
        target_lbl_name = target_path / "labelsTr" / (Path(lbl).name.replace("_mask", ""))
        if not target_lbl_name.exists():
            with TemporaryDirectory() as tempdir:
                img = sitk.ReadImage(lbl)
                img_array = sitk.GetArrayFromImage(img)
                # Remove the second label
                img_array[img_array == 2] = 0
                new_img = sitk.GetImageFromArray(img_array)
                new_img.CopyInformation(img)
                sitk.WriteImage(new_img, target_lbl_name)

    # ------------------------------- Dataset Json ------------------------------- #
    with open(get_dataset_path() / "Dataset501_hntsmrg_pre_primarytumor" / "dataset.json", "w") as f:
        json.dump(
            {
                "channel_names": {"0": "T2 MRI"},
                "labels": {"background": 0, "Primary Tumor": 1},
                "numTraining": 150,
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
