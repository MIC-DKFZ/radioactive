import json
from pathlib import Path
from tempfile import TemporaryDirectory

import nrrd
from tqdm import tqdm
from intrab.datasets_preprocessing.utils import (
    copy_images,
    copy_labels_of_modality_and_transform_to_instance,
)
from intrab.utils.io import get_dataset_path_by_id
import pandas as pd
import numpy as np
import SimpleITK as sitk
from toinstance import InstanceNrrd

from intrab.utils.paths import get_dataset_path


def download_ms_brain_dataset():
    data_url = "https://data.mendeley.com/datasets/8bctsm8jz7/1"


def main():
    """
    The BrainMetShare dataset is a collection of 3D MRI scans of brain metastases.
    Originally the dataset comes with four modalities, but the current models only take one.

    We create two versions of this dataset:
    - One using T1 gradient echo post-contrast modality.
    - One using T1 spin echo post-contrast modality.
    """

    # Need modality 0 (gradient echo post ce ) and 2 (spin echo post ce)
    data_path = get_dataset_path_by_id(201)

    target_ms_path = data_path.parent / "Dataset202_MS_Flair_instances"
    copy_images(data_path, target_ms_path)
    copy_labels_of_modality_and_transform_to_instance(
        data_path,
        target_ms_path,
        semantic_class_of_interest=1,
        dataset_json_description="D201 MS Flair derivative with instances of the lesion class.",
    )


def preprocess(raw_download_path: Path):
    """
    Two things we do:
    1. Images are already Nifi but have the wrong spacing set -- We respace them accordingly.
    2. Labels are not instances, so we transform them to instances.
    """
    # Load the meta data file
    data_dir = (
        raw_download_path
        / "Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information"
    )

    output_dir = get_dataset_path() / "Dataset201_MS_Flair_instances"
    image_target_path = output_dir / "imagesTr"
    image_target_path.mkdir(exist_ok=True, parents=True)
    label_target_path = output_dir / "labelsTr"
    label_target_path.mkdir(exist_ok=True, parents=True)

    meta_data = pd.read_excel(data_dir / "Supplementary Table 2 for  sequence parameters .xlsx", header=(0, 1))

    # Extract the "id" column and the "(FLAIR, Spacing between slices)" column
    id_column = np.array(meta_data[("Unnamed: 0_level_0", "ID")])
    spacing_column = np.array(meta_data[("FLAIR", "Spacing Between Slices")])

    cases = data_dir.glob("Patient*")

    for cur_case in tqdm(list(cases), desc="Processing MS FLAIR dataset", leave=False):
        case_id = int(cur_case.name.split("-")[-1])
        flair_path = cur_case / f"{case_id}-Flair.nii"
        flair_seg_path = cur_case / f"{case_id}-LesionSeg-Flair.nii"
        flair_img: sitk.Image = sitk.ReadImage(str(flair_path))
        flair_seg_img: sitk.Image = sitk.ReadImage(flair_seg_path)

        flair_spacing = flair_img.GetSpacing()
        # Change the through-plane spacing
        new_spacing = [flair_spacing[0], flair_spacing[1], float(spacing_column[id_column == case_id])]

        flair_img.SetSpacing(new_spacing)
        flair_seg_img.SetSpacing(new_spacing)

        with TemporaryDirectory() as tempdir:
            sitk.WriteImage(flair_seg_img, Path(tempdir) / "tmp_file.nii.gz")
            inrrd: InstanceNrrd = InstanceNrrd.from_semantic_img(
                Path(tempdir) / "tmp_file.nii.gz",
                do_cc=True,
                cc_kwargs={"dilation_kernel_radius": 0, "label_connectivity": 3},
            )
            array: list[np.ndarray] = [(cnt + 1) * arr for cnt, arr in enumerate(inrrd.get_instance_maps(class_id=1))]
            instance_array = np.sum(array, axis=0).astype(np.uint16)
            header = inrrd.get_vanilla_header()
            nrrd.write(str(Path(tempdir) / "tmp_file.nrrd"), instance_array, header)
            img = sitk.ReadImage(str(Path(tempdir) / "tmp_file.nrrd"))
            sitk.WriteImage(img, label_target_path / f"{case_id}_Flair.nii.gz")
        sitk.WriteImage(flair_img, image_target_path / f"{case_id}_Flair_0000.nii.gz")
    # ------------------------------- Dataset Json ------------------------------- #
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(
            {
                "channel_names": {"0": "FLAIR MRI"},
                "labels": {"background": 0, "MS lesion": 1},
                "numTraining": len(list(cases)),
                "file_ending": ".nrrd",
                "name": "D201 MS FLAIR",
                "reference": "https://data.mendeley.com/datasets/8bctsm8jz7/1",
                "description": "Holds instances of ms lesions from the MS Lesion FLAIR dataset.",
            },
            f,
        )


if __name__ == "__main__":
    main()
