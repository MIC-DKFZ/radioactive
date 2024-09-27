import os
from pathlib import Path
import nrrd
import numpy as np
import pydicom
from tqdm import tqdm
from intrab.datasets_preprocessing.conversion_utils import dicom_to_nrrd
from intrab.datasets_preprocessing.utils import suppress_output
from intrab.utils.paths import get_dataset_path
from toinstance import InstanceNrrd


def preprocess(raw_download_dir: Path):
    """Preprocessing code from Max Rokuss"""

    output_dir = get_dataset_path() / "Dataset913_adrenal_acc_ki67"

    dicom_dir = raw_download_dir / "adrenal_acc_ki67_seg"
    cases = sorted(os.listdir(str(dicom_dir)))
    matches = {}
    for case_id in tqdm(list(cases)):
        case_path = os.path.join(dicom_dir, case_id)
        subfolders = os.listdir(case_path)
        subfolder_with_seg = None
        if len(subfolders) != 1:
            folders_with_seg = 0
            for sub in subfolders:
                # Check if subfolders  have SEG
                scans = os.listdir(os.path.join(case_path, sub))
                seg_scans = [s for s in scans if "SEG" in s]
                if len(seg_scans) == 1:
                    folders_with_seg += 1
                    subfolder_with_seg = sub
                elif len(seg_scans) == 0:
                    continue
                else:
                    print(f"Case {case_id} has more than one segmentation!!!")
                    break
                if folders_with_seg > 1:
                    print(f"Case {case_id} has more than one segmentation!!!")
                    break
        if subfolder_with_seg is None:
            subfolder_with_seg = subfolders[0]
        scans = os.listdir(os.path.join(case_path, subfolder_with_seg))
        ct_scans = [s for s in scans if "CT" in s]
        seg_scans = [s for s in scans if "SEG" in s]
        if len(seg_scans) != 1:
            print(f"Case {case_id} has more than one segmentation!!!")
            continue

        # print(f"Case {case} has {len(ct_scans)} CT scans and 1 segmentation")
        # Load first slcie of seg
        seg_path = os.path.join(case_path, subfolder_with_seg, seg_scans[0])
        seg_path = os.path.join(seg_path, os.listdir(seg_path)[0])

        # load with pydicom
        seg = pydicom.dcmread(seg_path)
        seg_uid = seg.ReferencedSeriesSequence[0].SeriesInstanceUID

        for i, ct_scan in enumerate(ct_scans):
            path = os.path.join(case_path, subfolder_with_seg, ct_scan)
            path = os.path.join(path, os.listdir(path)[0])
            ct = pydicom.dcmread(path)
            # Get Name of scan
            if ct.SeriesInstanceUID == seg_uid:
                matches[case_id] = {"ct": path, "seg": seg_path}
                break

    # Go through matches and convert to nifi
    output_dir.mkdir(parents=True, exist_ok=True)
    imagesTr_dir: Path = output_dir / "imagesTr"
    labelsTr_dir: Path = output_dir / "labelsTr"
    imagesTr_dir.mkdir(parents=True, exist_ok=True)
    labelsTr_dir.mkdir(parents=True, exist_ok=True)

    for case_id, paths in matches.items():
        if (imagesTr_dir / (case_id + ".nrrd")).exists():
            continue

        ct: tuple[np.ndarray, dict]
        seg: tuple[np.ndarray, dict]

        ct_arr, ct_header = dicom_to_nrrd(paths["ct"])
        nrrd.write(str(imagesTr_dir / f"{case_id}_0000.nrrd"), ct_arr, ct_header)

        seg_arr, seg_header = dicom_to_nrrd(paths["seg"])

        try:
            if len(seg_arr.shape) == 4:
                seg_arr = seg_arr[1, :, :, :]
        except Exception as e:
            print(e)
            raise e

        if seg_arr.shape != ct_arr.shape:
            print(f"Case {case_id} has different shapes for CT and SEG")
            continue

        # ToDo: Add option to remove instance from innrrd through popping bin map values.
        innrrd: InstanceNrrd
        innrrd = InstanceNrrd.from_semantic_map(
            semantic_map=seg_arr,
            header=ct_header,  # We don't want to carry over 4D headers
            do_cc=True,
            cc_kwargs={"dilation_kernel_radius": 1, "label_connectivity": 3},
        )

        # binarize
        new_array = innrrd.array.copy()
        offset = 0
        for vals in np.unique(new_array):
            if vals != 0:
                n_samples = np.sum(new_array == vals)
                # Remove small components
                if n_samples < 30:
                    new_array[new_array == vals] = 0
                    offset += 1
                else:
                    new_array[new_array == vals] = vals - offset
        # Save the new array
        seg_innrrd: InstanceNrrd
        seg_innrrd = InstanceNrrd.from_instance_map(new_array, seg_header, class_name=1)
        seg_innrrd.to_file(labelsTr_dir / f"{case_id}.nrrd")
