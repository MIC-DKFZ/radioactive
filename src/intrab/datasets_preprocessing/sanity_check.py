from pathlib import Path
from toinstance import InstanceNrrd
import numpy as np
import nibabel as nib

from tqdm import tqdm


def check_groundtruths_are_mostly_contiguous(gt_path: Path, tolerance_voxels=10):
    if gt_path.name.endswith(".nrrd"):
        innrrd = InstanceNrrd.from_innrrd(gt_path)
        instance_maps = innrrd.get_instance_maps(1)
    else:
        img = nib.load(gt_path).get_fdata()
        instance_maps = [np.where(img == idx, 1, 0) for idx in np.unique(img) if idx != 0]

    for im in instance_maps:
        unique_zs = np.argwhere(im != 0)[:, 0]
        sorted_zs = np.sort(unique_zs)
        for i in range(1, len(sorted_zs)):
            if (sorted_zs[i] - sorted_zs[i - 1]) > tolerance_voxels:
                return False
    return True


def main():
    class_path = Path("/dkfz/cluster/gpu/data/OE0441/t006d/intra_bench/datasets")

    instance_classes_to_verify = [
        "Dataset201_MS_Flair_instances",
        # "Dataset209_hanseg_mr_oar",
        "Dataset501_hntsmrg_pre_primarytumor",
        # "Dataset600_pengwin",
        # "Dataset651_segrap",
        "Dataset911_LNQ_instances",
        "Dataset912_colorectal_livermets",
        "Dataset913_adrenal_acc_ki67",
        # "Dataset920_hcc_tace_liver",
        "Dataset921_hcc_tace_lesion",
        "Dataset930_RIDER_LungCT",
    ]

    for ds in instance_classes_to_verify:
        lbl_path = class_path / ds / "labelsTr"
        wrong_cases = []
        for img_path in tqdm(list(lbl_path.iterdir()), leave=False):
            if img_path.name.endswith((".nrrd", ".nii.gz")):
                passed = check_groundtruths_are_mostly_contiguous(img_path)
                if not passed:
                    wrong_cases.append(img_path.name)
        if len(wrong_cases) == 0:
            print(f"Dataset {ds}: Passed!")
        else:
            print(f"Dataset {ds}: Failed")
            print(wrong_cases)


if __name__ == "__main__":
    main()
