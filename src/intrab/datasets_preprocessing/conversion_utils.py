from pathlib import Path
from tempfile import TemporaryDirectory
import SimpleITK as sitk
from loguru import logger
import nrrd
import pydicom
from tqdm import tqdm
from intrab.utils.paths import get_MITK_path
import os
import numpy as np


def maybe_download_mitk():
    mitk_download_path = get_MITK_path()
    mitk_download_path.mkdir(exist_ok=True)
    # Download MITK
    download_url = (
        "https://www.mitk.org/download/releases/MITK-2024.06.2/Ubuntu%2022.04/MITK-v2024.06.2-linux-x86_64.tar.gz"
    )
    # wget the file
    mitk_tar = mitk_download_path / "MITK-v2024.06.2-linux-x86_64.tar.gz"
    if not (mitk_download_path / "MITK-v2024.06.2-linux-x86_64").exists():
        logger.info("Downloading MITK")
        os.system(f"wget {download_url} -P {mitk_download_path}")
        # Unzip the file

        os.system(f"tar -xvf {mitk_tar} -C {mitk_download_path}")
        # Remove the tar
    if mitk_tar.exists():
        os.system(f"rm {mitk_tar}")


def dicom_to_nrrd(dicom_series_path: Path) -> tuple[np.ndarray, dict]:
    """
    Takes a dicom series and converts it to a temporary nifti with MITK.
    Then reads it with SimpleITK and returns the image.
    """
    maybe_download_mitk()
    mitk_path = list(get_MITK_path().iterdir())[0] / "apps"  # Now in the MITK Snapshot folder
    dcm_sub_file = list(dicom_series_path.glob("*.dcm"))[0]
    with TemporaryDirectory() as temp_dir:
        os.system(f"cd {mitk_path} && ./MitkFileConverter.sh -i {dcm_sub_file} -o {temp_dir + "/tmp.nrrd"}")
        array, header = nrrd.read(temp_dir + "/tmp.nrrd")
    return array, header


def read_dicom_meta_data(dicom_folder: Path) -> dict:
    dicom_paths = list(dicom_folder.rglob("*.dcm"))[0]
    dicom = pydicom.dcmread(dicom_paths)
    return dicom


def get_matching_dicoms(dicoms: list[Path]) -> dict[str, dict[str, list[Path]]]:
    # Looks through a list of dicom folders and returns a dictionary with the matching CT and SEG dicoms.
    all_dicoms = {}
    for d in tqdm(dicoms, desc="Reading DICOM meta data", leave=False):
        meta_data = read_dicom_meta_data(d)
        study_name: str = meta_data.StudyInstanceUID
        modality = meta_data.Modality

        if study_name not in all_dicoms:
            all_dicoms[study_name] = {}

        if modality not in all_dicoms[study_name]:
            all_dicoms[study_name][modality] = []
        all_dicoms[study_name][modality].append(d)
    return all_dicoms
