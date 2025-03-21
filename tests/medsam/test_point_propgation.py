from pathlib import Path


def load_groundtruth_nifti() -> Path:
    some_path = "/path/to/groundtruth.nii.gz"
    return nib.load(some_path).get_fdata().astype(np.uint8)


def test_():
    assert True
