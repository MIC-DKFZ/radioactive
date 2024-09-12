from intrab.datasets_preprocessing.utils import (
    copy_files_of_modality,
    copy_labels_of_modality_and_transform_to_instance,
)
from intrab.utils.io import get_dataset_path_by_id


def main():
    """
    Transfer script moving the BrainMetShare dataset to the dataset format used in the benchmark.
    It expects the BrianMetShare dataset to be provided in nnU-Net dataformat with `channel_names` being:

    "channel_names": {
        "T1 gradient-echo post": 0,
        "T1 spin-echo pre": 1,
        "T1 spin-echo post": 2,
        "T2 FLAIR post images": 3
    }

    And labels being:
    "labels": {
        "background": 0,
        "enhancing": 1
    },

    The BrainMetShare dataset is a collection of 3D MRI scans of brain metastases collected of Stanford.

    We create two versions of this dataset:
    - One using T1 gradient echo post-contrast modality. (Modality 0000)
    - One using T1 spin echo post-contrast modality. (Modality 0002)
    """

    # Need modality 0 (gradient echo post ce ) and 2 (spin echo post ce)
    data_path = get_dataset_path_by_id(204)

    target_path_ge = data_path.parent / "Dataset205_stanford_brainemetshare_ge"
    target_path_se = data_path.parent / "Dataset206_stanford_brainemetshare_se"

    copy_files_of_modality(0, data_path, target_path_ge)
    copy_labels_of_modality_and_transform_to_instance(data_path, target_path_ge, 1)
    copy_files_of_modality(2, data_path, target_path_se)
    copy_labels_of_modality_and_transform_to_instance(data_path, target_path_se, 1)


if __name__ == "__main__":
    main()
