import json
import os
from pathlib import Path
from loguru import logger
import nrrd
import numpy as np
import pydicom
from tqdm import tqdm
from intrab.datasets_preprocessing.conversion_utils import (
    dicom_to_nrrd,
    nrrd_to_sitk,
    read_dicom_meta_data,
    resample_to_match,
    sitk_to_nrrd,
)
from intrab.datasets_preprocessing.utils import suppress_output
from intrab.utils.paths import get_dataset_path
from toinstance import InstanceNrrd
import SimpleITK as sitk
import pydicom_seg

exclude_cases = [
    # overlapping
    "CRLM-CT-1020",
    "CRLM-CT-1026",
    "CRLM-CT-1027",
    "CRLM-CT-1031",
    "CRLM-CT-1037",
    "CRLM-CT-1049",
    "CRLM-CT-1053",
    "CRLM-CT-1057",
    "CRLM-CT-1070",
    "CRLM-CT-1078",
    "CRLM-CT-1080",
    "CRLM-CT-1081",
    "CRLM-CT-1083",
    "CRLM-CT-1088",
    "CRLM-CT-1112",
    "CRLM-CT-1122",
    "CRLM-CT-1127",
    "CRLM-CT-1133",
    "CRLM-CT-1139",
    "CRLM-CT-1145",
    "CRLM-CT-1155",
    "CRLM-CT-1168",
    "CRLM-CT-1173",
    "CRLM-CT-1186",
    "CRLM-CT-1190",
    # missing data (at least with my download :) )
    "CRLM-CT-1183",
]


def select_folder_from_directory(directory: Path) -> Path:
    all_dirs = [p for p in directory.iterdir() if p.is_dir()]
    assert len(all_dirs) == 1, f"Expected one directory in {directory} but found {all_dirs}"
    return all_dirs[0]


def assert_coord_sys(img0_itk: sitk.Image, img1_itk: sitk.Image):
    assert img0_itk.GetSpacing() == img1_itk.GetSpacing()
    assert img0_itk.GetOrigin() == img1_itk.GetOrigin()
    assert img0_itk.GetDirection() == img1_itk.GetDirection()


def preprocess(raw_download_dir: Path):
    """Preprocessing code from Max Rokuss"""

    output_dir = get_dataset_path() / "Dataset912_colorectal_livermets"

    dicom_dir = raw_download_dir / "colorectal_liver_metastases"
    cases = sorted(os.listdir(str(dicom_dir)))
    assert len(cases) == 197, f"Expected 197 cases but found {len(cases)}"
    cases = [c for c in cases if c not in exclude_cases]
    logger.info(f"Processing not-excluded {len(cases)} cases")
    matches = {}

    for case_id in tqdm(list(cases)):
        case_path = dicom_dir / case_id
        series_dir = list(case_path.iterdir())[0]

        # Check if subfolders  have SEG
        seg_scans = [s for s in series_dir.iterdir() if "SEG" in s.name]
        seg_metainfo = [read_dicom_meta_data(s) for s in seg_scans]
        seg_scan = [s for s, meta in zip(seg_scans, seg_metainfo) if meta.SeriesDescription == "Segmentation"]
        assert len(seg_scan) == 1, "More than one segmentation found"
        seg_scan = seg_scan[0]

        ct_scan = [s for s in series_dir.iterdir() if "CT" in s.name]
        assert len(ct_scan) == 1, "More than one CT scan found"
        ct_scan = ct_scan[0]

        matches[case_id] = {"ct": ct_scan, "seg": seg_scan}

    # Go through matches and convert to nifi
    output_dir.mkdir(parents=True, exist_ok=True)
    imagesTr_dir: Path = output_dir / "imagesTr"
    labelsTr_dir: Path = output_dir / "labelsTr"
    imagesTr_dir.mkdir(parents=True, exist_ok=True)
    labelsTr_dir.mkdir(parents=True, exist_ok=True)

    for case_id, paths in matches.items():
        if (imagesTr_dir / (case_id + ".nrrd")).exists():
            continue

        ct_arr: np.ndarray
        ct_header: dict
        seg_arr: np.ndarray
        seg_header: dict

        ct_arr, ct_header = dicom_to_nrrd(paths["ct"])
        nrrd.write(str(imagesTr_dir / f"{case_id}_0000.nrrd"), ct_arr, ct_header)

        seg_arr, seg_header = dicom_to_nrrd(paths["seg"])
        nrrd.write(str(labelsTr_dir / f"orig_{case_id}.nrrd"), seg_arr, seg_header)

        keys_in_labels: dict[str, int] = {
            group["labels"][0]["name"]: cnt
            for cnt, group in enumerate(json.loads(seg_header["org.mitk.multilabel.segmentation.labelgroups"]))
        }
        keys_of_interest = [v for k, v in keys_in_labels.items() if k.startswith("Tumor")]

        tumor_instance_maps = list(np.where(seg_arr[keys_of_interest] != 0, 1, 0))

        innrrd: InstanceNrrd
        innrrd = InstanceNrrd.from_binary_instance_maps(
            instance_dict={1: tumor_instance_maps},
            header=seg_header,  # We don't want to carry over 4D headers
            maps_mutually_exclusive=True,
        )
        seg_arr = innrrd.array
        seg_header = innrrd.header

        if seg_arr.shape != ct_arr.shape:
            ct_img: sitk.Image = nrrd_to_sitk(ct_arr, ct_header)
            seg_img: sitk.Image = nrrd_to_sitk(innrrd.array, innrrd.header)
            resampled_seg_img = resample_to_match(reference_img=ct_img, resample_img=seg_img, is_seg=True)
            seg_arr, seg_header = sitk_to_nrrd(resampled_seg_img)

        nrrd.write(str(labelsTr_dir / f"{case_id}.nrrd"), seg_arr, seg_header)
    # ------------------------------- Dataset Json ------------------------------- #
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(
            {
                "channel_names": {"0": "CT"},
                "labels": {"background": 0, "lesion": 1},
                "numTraining": len(list(cases)),
                "file_ending": ".nrrd",
                "name": "d912 Colorectal Liver Metastases",
                "description": "Dataset sources from colorectal_liver_metastases dataset from TCIA. Only using Tumor instances.",
            },
            f,
        )
