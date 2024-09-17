from pathlib import Path
from tempfile import TemporaryDirectory
import SimpleITK as sitk
from loguru import logger
import nrrd
import pydicom
from tqdm import tqdm
from intrab.datasets_preprocessing.utils import suppress_output
from intrab.utils.paths import get_MITK_path
import os
import nibabel as nib
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


def nrrd_to_sitk(nrrd_arr, nrrd_header) -> sitk.Image:
    """Converts an nrrd array to an sitk Image."""
    with TemporaryDirectory() as temp_dir:
        nrrd.write(temp_dir + "/tmp.nrrd", nrrd_arr, nrrd_header)
        image = sitk.ReadImage(temp_dir + "/tmp.nrrd")
        # Need to make sure image is in memory before deleting
        image_arr = sitk.GetArrayFromImage(image)
        new_img = sitk.GetImageFromArray(image_arr)
        new_img.CopyInformation(image)
    return new_img


def load_any_to_nib(image_path: Path | str) -> nib.Nifti1Image:
    """Loads an image in any format and returns a nibabel image."""
    image_path = Path(image_path)
    read_im = sitk.ReadImage(image_path)
    with TemporaryDirectory() as temp_dir:
        sitk.WriteImage(read_im, temp_dir + "/temp.nii.gz")
        image = nib.load(temp_dir + "/temp.nii.gz")
        image.get_fdata()  # Need to make sure image is in memory before deleting
    return image


def nrrd_to_nib(nrrd_arr, nrrd_header) -> nib.Nifti1Image:
    """Converts an nrrd array to a nibabel Image."""
    with TemporaryDirectory() as temp_dir:
        nrrd.write(temp_dir + "/tmp.nrrd", nrrd_arr, nrrd_header)
        img = sitk.ReadImage(temp_dir + "/tmp.nrrd")
        sitk.WriteImage(img, temp_dir + "/tmp.nii")
        image: nib.Nifti1Image = nib.load(temp_dir + "/tmp.nii")
        _ = image.get_fdata()  # Need to make sure image is in memory before deleting
    return image


def sitk_to_nrrd(sitk_img: sitk.Image) -> tuple[np.ndarray, dict]:
    """Converts a sitk image to an nrrd array."""
    with TemporaryDirectory() as temp_dir:
        sitk.WriteImage(sitk_img, temp_dir + "/tmp.nrrd")
        nrrd_arr, nrrd_header = nrrd.read(temp_dir + "/tmp.nrrd")
    return nrrd_arr, nrrd_header


def dicom_to_nrrd(dicom_series_path: Path | str) -> tuple[np.ndarray, dict]:
    """
    Takes a dicom series and converts it to a temporary nifti with MITK.
    Then reads it with SimpleITK and returns the image.
    """
    with suppress_output():
        dicom_series_path = Path(dicom_series_path)
        dicom_series_path = Path(dicom_series_path)
        maybe_download_mitk()
        mitk_path = list(get_MITK_path().iterdir())[0] / "apps"  # Now in the MITK Snapshot folder
        if dicom_series_path.is_dir():
            dcm_sub_file = list(dicom_series_path.glob("*.dcm"))[0]
        else:
            dcm_sub_file = dicom_series_path
        with TemporaryDirectory() as temp_dir:
            os.system(f"cd {mitk_path} && ./MitkFileConverter.sh -i {dcm_sub_file} -o {temp_dir + "/tmp.nrrd"}")
            array, header = nrrd.read(temp_dir + "/tmp.nrrd")
    return array, header


def read_dicom_meta_data(dicom_folder: Path) -> dict:
    """Reads the meta data from a DICOM file or from one image of a Series."""
    if dicom_folder.is_dir():
        dicom_paths = list(dicom_folder.glob("*.dcm"))[0]
    else:
        dicom_paths = dicom_folder
    dicom = pydicom.dcmread(dicom_paths)
    return dicom


def get_dicoms_meta_info(dicoms: list[Path]) -> dict[str, dict[str, list[Path]]]:
    # Looks through a list of dicom folders and returns a dictionary with the matching CT and SEG dicoms.
    all_dicoms = {}
    for d in tqdm(dicoms, desc="Reading DICOM meta data", leave=False):
        meta_data = read_dicom_meta_data(d)
        study_name: str = meta_data.StudyInstanceUID
        modality = meta_data.Modality

        content = {"filepath": d, "PatientID": meta_data.PatientID, "StudyInstanceUID": study_name}

        if study_name not in all_dicoms:
            all_dicoms[study_name] = {}

        if modality not in all_dicoms[study_name]:
            all_dicoms[study_name][modality] = []
        if modality == "SEG":
            content["reference_series"] = meta_data.ReferencedSeriesSequence[0].SeriesInstanceUID

        all_dicoms[study_name][modality].append(content)
    return all_dicoms


def get_matching_img(img_dicom_meta_info: list[dict], seg_dicom_meta_info: dict) -> dict:
    """
    Finds the reference image for the segmentation and returns the matching image UUID.
    """
    # Matches the image to the segmentation
    for img in img_dicom_meta_info:
        im_uid = seg_dicom_meta_info["reference_series"]
        if img["filepath"] == im_uid:
            return img
    return None


def resample_to_match(reference_img: sitk.Image, resample_img: sitk.Image, is_seg: bool) -> sitk.Image:
    """
    Resample the target image to match the reference image's size, spacing, origin, and direction.

    Args:
    - reference_img (sitk.Image): The image whose properties (size, spacing, origin, direction) you want to match.
    - target_img (sitk.Image): The image to be resampled to match the reference.

    Returns:
    - sitk.Image: The resampled target image.
    """
    # Create the resample filter
    resample = sitk.ResampleImageFilter()

    # Set reference properties: size, spacing, origin, and direction
    resample.SetSize(reference_img.GetSize())
    resample.SetOutputSpacing(reference_img.GetSpacing())
    resample.SetOutputOrigin(reference_img.GetOrigin())
    resample.SetOutputDirection(reference_img.GetDirection())

    # Set the interpolator - use nearest neighbor for label images or linear for continuous data
    if is_seg:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)  # Change to sitkNearestNeighbor if working with segmentation

    # Set the default pixel value for any padding area (e.g., 0 for background)
    resample.SetDefaultPixelValue(0)

    # Resample the target image
    resampled_img = resample.Execute(resample_img)

    return resampled_img
