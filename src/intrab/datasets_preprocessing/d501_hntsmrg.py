import json
import os
from pathlib import Path
import shutil
from toinstance import InstanceNrrd

from tqdm import tqdm

from intrab.datasets_preprocessing.utils import (
    copy_images,
    copy_labels_of_modality_and_transform_to_instance,
    download_from_zenodo,
)
from intrab.utils.io import get_dataset_path_by_id


def dataset_download():
    zenodo_id = "11199559"


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
