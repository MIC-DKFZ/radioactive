from copy import deepcopy
import json
from pathlib import Path

import nrrd
import numpy as np
from tqdm import tqdm

from intrab.datasets_preprocessing.conversion_utils import (
    dicom_to_nrrd,
    get_dicoms_meta_info,
    nrrd_to_sitk,
    resample_to_match,
    sitk_to_nrrd,
)
from intrab.utils.paths import get_dataset_path
from toinstance import InstanceNrrd
import SimpleITK as sitk


def preprocess(raw_download_dir: Path):
    dicoms_ct = [p for p in list((raw_download_dir / "RIDER-Lung-CT-Scans").iterdir()) if p.is_dir()]
    dicoms_seg = [p for p in list((raw_download_dir / "RIDER-Lung-CT-Seg").iterdir()) if p.is_dir()]

    output_dir = get_dataset_path() / "Dataset930_RIDER_LungCT"
    images_dir = output_dir / "imagesTr"
    labels_dir = output_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Get Image Label Pairs
    all_dicoms = get_dicoms_meta_info(dicoms_ct + dicoms_seg)
    all_dicoms = {key: all_dicoms[key] for key in sorted(all_dicoms)}
    cnt_dicom_map = {}

    for cnt, (study_name, dicom_data) in tqdm(
        enumerate(all_dicoms.items()), total=len(all_dicoms), desc="Converting RIDER DICOMs to NRRD"
    ):
        ct_path = dicom_data["CT"][0]["filepath"]  # We only have one CT and one SEG in LNQ
        seg_path = dicom_data["SEG"][0]["filepath"]
        ct: tuple[np.ndarray, dict] = dicom_to_nrrd(ct_path)
        seg: tuple[np.ndarray, dict] = dicom_to_nrrd(seg_path)

        if len(seg[0].shape) == 5:
            print("Wait")
        elif len(seg[0].shape) == 4:
            # If the shape is 4D the first dimension are predictions and the second are the GT
            # So we overwrite the final_seg content.
            seg = (seg[0][1], InstanceNrrd.clean_header(seg[1]))
        else:
            print("Unexpected")

        if ct[0].shape != seg[0].shape:
            print(f"Shape mismatch: {ct[0].shape} != {seg[0].shape}")
            ct_img: sitk.Image = nrrd_to_sitk(*ct)
            seg_img: sitk.Image = nrrd_to_sitk(*seg)
            resampled_seg_img = resample_to_match(reference_img=ct_img, resample_img=seg_img, is_seg=True)
            seg = sitk_to_nrrd(resampled_seg_img)
        # nrrd.write(str(labels_dir / f"original_seg_rider_lung_{cnt:04d}.nrrd"), seg[0], seg[1])

        # If the shape is 4D the first dimension are predictions and the second are the GT

        # temporary save the CT as the header is not the same as the SEG

        seg = (seg[0], ct[1])  # Copy the header from the CT to the SEG

        innrrd = InstanceNrrd.from_semantic_map(
            semantic_map=seg[0],
            header=deepcopy(seg[1]),
            do_cc=True,
            cc_kwargs={"dilation_kernel_radius": 0, "label_connectivity": 3},
        )
        innrrd.to_file(labels_dir / f"rider_lung_{cnt:04d}.nrrd")
        nrrd.write(str(images_dir / f"rider_lung_{cnt:04d}_0000.nrrd"), ct[0], ct[1])
        cnt_dicom_map[cnt] = study_name
    # ------------------------------- Dataset Json ------------------------------- #
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(
            {
                "channel_names": {"0": "CT"},
                "labels": {"background": 0, "lesion": 1},
                "numTraining": len(list(all_dicoms)),
                "file_ending": ".nrrd",
                "name": "d911 LNQ instance lesions",
                "dataset_type": "instance",
            },
            f,
        )
    with open(output_dir / "study_name_patient_id_map.json", "w") as f:
        json.dump(cnt_dicom_map, f)

    print(all_dicoms)
