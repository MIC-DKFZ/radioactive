import json
import os
from pathlib import Path
import shutil
import subprocess
from loguru import logger
from toinstance import InstanceNrrd
import numpy as np
import nrrd
from tqdm import tqdm
from pydrive.drive import GoogleDrive, GoogleDriveFileList
from pydrive.auth import GoogleAuth
import zipfile
import gdown
import SimpleITK as sitk


from intrab.utils.paths import get_dataset_path

GRDRIVE_FILE_LIST = {
    "SegRap2023_Training_Set_120cases.zip": "1lyBSgAyfKb6USjCqyxwnb72KaSf37r14",
    "SegRap2023_Training_Set_120cases_OneHot_Labels.zip": "1Zl-Y3jAzEwlkAXbNCRAUbGSRznPh3rkA",
    "SegRap2023_Training_Set_120cases_Update_Labels(Task001).zip": "1heL4-11mBbARrDYaZqyJTAK6LjHS60IH",
    "convert_one_hot_labels_to_multi_organs.py": "1uCmqbrowmFVbXHkhtuYTXceZx_PGIalv",
    "dataset_task001.json": "1kUyOdLMOF2stERvka74byXJzgWYc5ZUU",
    "oar_gtv_name.json": "1UOs4k5MTEPDeLVRNmQzx3Fbk9i4s_Gi-",
}

SEGRAP_TASK001_ONE_HOT_LABEL_NAMES = {
    "Brain": 1,
    "BrainStem": 2,
    "Chiasm": 3,
    "TemporalLobe_L": 4,
    "TemporalLobe_R": 5,
    "TemporalLobe_Hippocampus_OverLap_L": 6,
    "TemporalLobe_Hippocampus_OverLap_R": 7,
    "Hippocampus_L": 8,
    "Hippocampus_R": 9,
    "Eye_L": 10,
    "Eye_R": 11,
    "Lens_L": 12,
    "Lens_R": 13,
    "OpticNerve_L": 14,
    "OpticNerve_R": 15,
    "MiddleEar_L": 16,
    "MiddleEar_R": 17,
    "IAC_L": 18,
    "IAC_R": 19,
    "MiddleEar_TympanicCavity_OverLap_L": 20,
    "MiddleEar_TympanicCavity_OverLap_R": 21,
    "TympanicCavity_L": 22,
    "TympanicCavity_R": 23,
    "MiddleEar_VestibulSemi_OverLap_L": 24,
    "MiddleEar_VestibulSemi_OverLap_R": 25,
    "VestibulSemi_L": 26,
    "VestibulSemi_R": 27,
    "Cochlea_L": 28,
    "Cochlea_R": 29,
    "MiddleEar_ETbone_OverLap_L": 30,
    "MiddleEar_ETbone_OverLap_R": 31,
    "ETbone_L": 32,
    "ETbone_R": 33,
    "Pituitary": 34,
    "OralCavity": 35,
    "Mandible_L": 36,
    "Mandible_R": 37,
    "Submandibular_L": 38,
    "Submandibular_R": 39,
    "Parotid_L": 40,
    "Parotid_R": 41,
    "Mastoid_L": 42,
    "Mastoid_R": 43,
    "TMjoint_L": 44,
    "TMjoint_R": 45,
    "SpinalCord": 46,
    "Esophagus": 47,
    "Larynx": 48,
    "Larynx_Glottic": 49,
    "Larynx_Supraglot": 50,
    "Larynx_PharynxConst_OverLap": 51,
    "PharynxConst": 52,
    "Thyroid": 53,
    "Trachea": 54,
}

SEGRAP_SUBSETS = {
    "Brain": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "BrainStem": 2,
    "Chiasm": 3,
    "TemporalLobe_L": [4, 6],
    "TemporalLobe_R": [5, 7],
    "Hippocampus_L": [8, 6],
    "Hippocampus_R": [9, 7],
    "Eye_L": [10, 12],
    "Eye_R": [11, 13],
    "Lens_L": 12,
    "Lens_R": 13,
    "OpticNerve_L": 14,
    "OpticNerve_R": 15,
    "MiddleEar_L": [18, 16, 20, 24, 28, 30],
    "MiddleEar_R": [19, 17, 21, 25, 29, 31],
    "IAC_L": 18,
    "IAC_R": 19,
    "TympanicCavity_L": [22, 20],
    "TympanicCavity_R": [23, 21],
    "VestibulSemi_L": [26, 24],
    "VestibulSemi_R": [27, 25],
    "Cochlea_L": 28,
    "Cochlea_R": 29,
    "ETbone_L": [32, 30],
    "ETbone_R": [33, 31],
    "Pituitary": 34,
    "OralCavity": 35,
    "Mandible_L": 36,
    "Mandible_R": 37,
    "Submandibular_L": 38,
    "Submandibular_R": 39,
    "Parotid_L": 40,
    "Parotid_R": 41,
    "Mastoid_L": 42,
    "Mastoid_R": 43,
    "TMjoint_L": 44,
    "TMjoint_R": 45,
    "SpinalCord": 46,
    "Esophagus": 47,
    "Larynx": [48, 49, 50, 51],
    "Larynx_Glottic": 49,
    "Larynx_Supraglot": 50,
    "PharynxConst": [51, 52],
    "Thyroid": 53,
    "Trachea": 54,
}


def download_from_gdrive(file_id, filename):
    # Step 1: Get the initial confirmation token (if needed) using wget
    command_1 = f"wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id={file_id}' -O-"
    result = subprocess.run(command_1, shell=True, capture_output=True, text=True)

    # Step 2: Extract the confirmation token
    confirm_token = None
    for line in result.stdout.splitlines():
        if "confirm=" in line:
            confirm_token = line.split("confirm=")[1].split("&")[0]
            break

    # Step 3: Download the file with the confirmation token (if token is found)
    if confirm_token:
        command_2 = f"wget --load-cookies cookies.txt 'https://docs.google.com/uc?export=download&confirm={confirm_token}&id={file_id}' -O {filename}"
    else:
        # If no confirmation token is needed (small file), proceed to download
        command_2 = (
            f"wget --load-cookies cookies.txt 'https://docs.google.com/uc?export=download&id={file_id}' -O {filename}"
        )

    subprocess.run(command_2, shell=True)

    # Step 4: Clean up the cookies.txt file
    subprocess.run("rm -f cookies.txt", shell=True)


def create_label_instance_nrrd_for_segrap_case(case_dir: Path) -> tuple[InstanceNrrd, dict]:
    """
    Creates label maps of regions for certain organs.
    """

    header: dict = None
    res_innrd_labelmap: list[dict] = []
    for img in case_dir.iterdir():
        class_name = img.name.replace(".nii.gz", "")
        for cnt, segrap_str in enumerate(SEGRAP_SUBSETS):
            if segrap_str == class_name:
                img_data, header = nrrd.read(img)
                res = {"data": img_data, "label_name": class_name, "label_id": cnt + 1}
                res_innrd_labelmap.append(res)

    # This is quite redundant as it gets created 42x but who cares, it's run once.
    dataset_json = {
        "channel_names": {"0": "T1 MRI"},
        "labels": {k["label_name"]: k["label_id"] for k in res_innrd_labelmap},
        "numTraining": 42,
        "file_ending": ".nrrd",
        "name": "Dataset651_segrap_mr",
        "description": "Dataset containing the semantic regions of the head and neck, as defined by the SEGRAP task.",
    }

    innrrd_binmaps = {l["label_id"]: [l["data"]] for l in res_innrd_labelmap}

    return InstanceNrrd.from_binary_instance_maps(innrrd_binmaps, header), dataset_json


def main():
    temp_segrap_path = download_segrap_dataset()

    cases_dir = temp_segrap_path / "SegRap2023_Training_Set_120cases"
    assert (cases_dir).exists(), f"Directory {cases_dir} does not contain the expected structure."
    f: Path
    # assert (
    #     len([f for f in cases_dir.iterdir() if f.is_dir()]) == 42
    # ), f"Directory {hanseg_dir} does not contain 42 cases."

    output_dir = get_dataset_path() / "Dataset651_segrap_mr"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------- Do MR Images ------------------------------- #
    output_image_dir = output_dir / "imagesTr"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    for case_dir in tqdm(list(cases_dir.iterdir()), desc="Copying MR Images"):
        if case_dir.name.startswith("segrap"):
            mr_img = case_dir / "image.nii.gz"  # OR: image_contrast.nii.gz
            sitk_im = sitk.ReadImage(str(mr_img))
            sitk.WriteImage(sitk_im, str(output_image_dir / f"{case_dir.name}_0000.nrrd"))

    # --------------------------------- Do Labels -------------------------------- #
    output_label_dir = output_dir / "labelsTr"
    output_label_dir.mkdir(parents=True, exist_ok=True)
    for case_dir in tqdm(list(cases_dir.iterdir()), desc="Creating Labels"):
        if case_dir.name.startswith("segrap"):
            instance_nrrd, dataset_json = create_label_instance_nrrd_for_segrap_case(case_dir)
            instance_nrrd.to_file(output_label_dir / f"{case_dir.name}.in.nrrd")

    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f)


if __name__ == "__main__":
    main()
