from argparse import Namespace
import os
import json
from datetime import datetime

from intrab.experiments_2d import run_experiments_2d
from intrab.model.SAM import SAMInferer
from intrab.model.SAMMed2D import SAMMed2DInferer
from intrab.model.MedSAM import MedSAMInferer
from intrab.model.SAMMed3D import SAMMed3DInferer

inferer_registry = {
    "sam": SAMInferer,
    "sammed2d": SAMMed2DInferer,
    "medsam": MedSAMInferer,
    "sammed3d": SAMMed3DInferer,
}


def get_img_gts_jhu(dataset_dir):
    images_dir = os.path.join(dataset_dir, "imagesTr")
    labels_dir = os.path.join(dataset_dir, "labelsTr")
    imgs_gts = [
        (
            os.path.join(images_dir, img_path),
            os.path.join(labels_dir, img_path.removesuffix("_0000.nii.gz") + ".nii.gz"),
        )
        for img_path in os.listdir(images_dir)  # Adjust the extension as needed
        if os.path.exists(os.path.join(labels_dir, img_path.rstrip("_0000.nii.gz") + ".nii.gz"))
    ]
    return imgs_gts


def get_imgs_gts_amos(dataset_dir):
    images_dir = os.path.join(dataset_dir, "imagesTs")
    labels_dir = os.path.join(dataset_dir, "labelsTs")
    imgs_gts = [
        (os.path.join(images_dir, img_path), os.path.join(labels_dir, os.path.basename(img_path)))
        for img_path in os.listdir(images_dir)  # Adjust the extension as needed
        if os.path.exists(os.path.join(labels_dir, os.path.basename(img_path)))
    ]
    return imgs_gts


def get_imgs_gts_segrap(dataset_dir):
    images_dir = os.path.join(dataset_dir, "imagesTr")
    labels_dir = os.path.join(dataset_dir, "labelsTr")
    imgs_gts = [
        (
            os.path.join(images_dir, img_path),
            os.path.join(labels_dir, img_path.removesuffix("_0000.nii.gz") + ".nii.gz"),
        )
        for img_path in os.listdir(images_dir)  # Adjust the extension as needed
        if os.path.exists(os.path.join(labels_dir, img_path.removesuffix("_0000.nii.gz") + ".nii.gz"))
    ]
    return imgs_gts


checkpoint_registry = {
    "sam": "/home/t722s/Desktop/UniversalModels/TrainedModels/sam_vit_h_4b8939.pth",
    "medsam": "/home/t722s/Desktop/UniversalModels/TrainedModels/medsam_vit_b.pth",
    "sammed2d": "/home/t722s/Desktop/UniversalModels/TrainedModels/sam-med2d_b.pth",
}

dataset_registry = {
    "abdomenAtlas": {
        "dir": "/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_2img/",
        "dataset_func": get_img_gts_jhu,
    },
    "segrap": {"dir": "/home/t722s/Desktop/Datasets/segrapSub/", "dataset_func": get_imgs_gts_segrap},
}

if __name__ == "__main__":
    # Setup
    # warnings.filterwarnings('error')

    dataset_name = "abdomenAtlas"
    model_name = "medsam"
    results_dir = "/home/t722s/Desktop/ExperimentResults"

    exp_params = Namespace(
        n_click_random_points=5,
        n_slice_point_interpolation=5,
        n_slice_box_interpolation=5,
        n_seed_points_point_propagation=5,
        n_points_propagation=5,
        dof_bound=60,
        perf_bound=0.85,
    )  # Todo: Make this a dataclass instead, so attributes are transparently available
    device = "cuda"
    seed = 11121
    label_overwrite = None
    experiment_overwrite = None

    # prompt_types = ['points', 'boxes', 'interactive']
    prompt_types = ["boxes"]

    label_overwrite = {
        "kidney_left": 3,
    }

    # label_overwrite = {
    #     "background": 0,
    #     "aorta": 1,
    #     "gall_bladder": 2,
    #     "kidney_left": 3,
    #     "kidney_right": 4,
    #     "liver": 5,
    #     "pancreas": 6,
    #     "postcava": 7,
    #     "spleen": 8,
    #     "stomach": 9
    # }

    # experiment_overwrite = ['random_points']

    # Get (img path, gt path) pairs
    results_path = os.path.join(
        results_dir, model_name + "_" + dataset_name + "_" + datetime.now().strftime("%Y%m%d_%H%M")
    )
    dataset_func, dataset_dir = dataset_registry[dataset_name]["dataset_func"], dataset_registry[dataset_name]["dir"]
    imgs_gts = dataset_func(dataset_dir)

    # Get dataset dict if missing
    with open(os.path.join(dataset_dir, "dataset.json"), "r") as f:
        dataset_info = json.load(f)
    label_dict = dataset_info["labels"]

    if label_overwrite:
        label_dict = label_overwrite

    # Load the model
    checkpoint_path = checkpoint_registry[model_name]
    inferer = inferer_registry[model_name](checkpoint_path, device)

    # Run experiments
    run_experiments_2d(
        inferer,
        imgs_gts,
        results_path,
        label_dict,
        exp_params,
        prompt_types,
        seed=1,
        experiment_overwrite=experiment_overwrite,
        save_segs=True,
    )
