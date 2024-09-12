from intrab.datasets_preprocessing.utils import (
    copy_images,
    copy_labels_of_modality_and_transform_to_instance,
)
from intrab.utils.io import get_dataset_path_by_id


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


if __name__ == "__main__":
    main()
