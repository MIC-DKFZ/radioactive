from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from loguru import logger
import pydicom
import SimpleITK as sitk
import numpy as np
import nrrd
from toinstance import InstanceNrrd
from tqdm import tqdm

from intrab.datasets_preprocessing.conversion_utils import dicom_to_nrrd, get_matching_dicoms
from intrab.utils.paths import get_dataset_path


def read_dicom_series_with_metadata(dicom_folder, output_file) -> tuple[np.ndarray, dict]:
    """
    Reads a DICOM series from a folder, retains metadata, and saves it as a single file (NIfTI or NRRD).

    Args:
    - dicom_folder (str): Path to the folder containing DICOM files.
    - output_file (str): Path to save the output file (with .nii or .nrrd extension).
    - output_format (str): Format of the output file ("nifti" or "nrrd").
    """
    # Create a reader object to read the DICOM series
    reader = sitk.ImageSeriesReader()

    # Get the file names of the DICOM series
    dicom_series = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_series)

    # Read the DICOM series into a SimpleITK image
    image = reader.Execute()

    # Copy DICOM tags from the first slice (you can use other metadata if needed)
    meta_data = reader.GetMetaData()
    for tag in reader.GetMetaDataKeys(0):
        value = reader.GetMetaData(0, tag)
        image.SetMetaData(tag, value)

    # Save as NIfTI or NRRD, retaining the metadata
    with TemporaryDirectory() as temp_dir:
        sitk.WriteImage(image, temp_dir + "/tmp.nrrd")
        nrrd.read(temp_dir + "/tmp.nrrd")
        sitk.WriteImage(image, output_file + ".nrrd")


def preprocess(raw_download_dir: Path):
    dicoms = [p for p in list(raw_download_dir.iterdir()) if p.is_dir()]

    output_dir = get_dataset_path() / "Dataset911_LNQ_instances"
    images_dir = output_dir / "imagesTr"
    labels_dir = output_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Get Image Label Pairs
    all_dicoms = get_matching_dicoms(dicoms)
    all_dicoms = {key: all_dicoms[key] for key in sorted(all_dicoms)}
    cnt_dicom_map = {}

    for cnt, (study_name, dicom_data) in tqdm(enumerate(all_dicoms.items()), desc="Converting LNQ DICOMs to NRRD"):
        ct_path = dicom_data["CT"][0]  # We only have one CT and one SEG in LNQ
        seg_path = dicom_data["SEG"][0]
        ct: tuple[np.ndarray, dict] = dicom_to_nrrd(ct_path)
        seg: tuple[np.ndarray, dict] = dicom_to_nrrd(seg_path)

        innrrd = InstanceNrrd.from_semantic_map(
            semantic_map=seg[0],
            header=deepcopy(seg[1]),
            do_cc=True,
            cc_kwargs={"dilation_kernel_radius": 0, "label_connectivity": 3},
        )
        innrrd.to_file(labels_dir / f"lnq_{cnt:04d}.nrrd")
        nrrd.write(str(images_dir / f"lnq_{cnt:04d}_0000.nrrd"), ct[0], ct[1])
        cnt_dicom_map[cnt] = study_name
    # ------------------------------- Dataset Json ------------------------------- #
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(
            {
                "channel_names": {"0": "T1 MRI"},
                "labels": {"background": 0, "lesion": 1},
                "numTraining": len(list(all_dicoms)),
                "file_ending": ".nrrd",
                "name": "d911 LNQ instance lesions",
            },
            f,
        )
    with open(output_dir / "study_name_patient_id_map.json", "w") as f:
        json.dump(cnt_dicom_map, f)

    print(all_dicoms)
