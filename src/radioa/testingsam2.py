# Imports and set up inferer
import nibabel as nib
import os

os.environ["INTRAB_MODEL_PATH"] = "/home/t722s/E132-Projekte/Projects/2023_Tempus_intrab/checkpoints"

from radioa.model import SAM2
from radioa.model.model_utils import checkpoint_registry
from radioa.utils.io import binarize_gt
from importlib import reload
import numpy as np

# Model things
img_path = "/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_2img/imagesTr/BDMAP_00000001_0000.nii.gz"
gt_path = "/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_2img/labelsTr/BDMAP_00000001.nii.gz"
# img_path = '/home/t722s/cluster-data_all/t006d/intra_bench/datasets/Dataset209_hanseg_mr_oar/imagesTr/case_01_0000.nrrd'
# gt_path = '/home/t722s/cluster-data_all/t006d/intra_bench/datasets/Dataset209_hanseg_mr_oar/labelsTr/case_01.nrrd'
# img_path = '/home/t722s/cluster-data_all/t006d/intra_bench/datasets/Dataset201_MS_Flair_instances/imagesTr/2_Flair_0000.nrrd'
# gt_path = '/home/t722s/cluster-data_all/t006d/intra_bench/datasets/Dataset201_MS_Flair_instances/labelsTr/2_Flair.nrrd'
checkpoint = checkpoint_registry["sam2"]
device = "cuda"

inferer = SAM2.SAM2Inferer(checkpoint, device)

target_label = 3
binary_gt_orig_coords = binarize_gt(gt_path, target_label)


import radioa.prompts.interactive_prompter as i
import radioa.prompts.prompter as p

from radioa.prompts.prompt_hparams import PromptConfig

exp_params = PromptConfig(
    twoD_n_click_random_points=5,
    twoD_n_slice_point_interpolation=5,
    twoD_n_slice_box_interpolation=5,
    twoD_n_seed_points_point_propagation=5,
    twoD_n_points_propagation=5,
    # interactive_dof_bound=60,
    # interactive_perf_bound=0.9,
    # interactive_max_iter=10,
    interactive_max_iter=5,
    twoD_interactive_n_cc=1,
    twoD_interactive_n_points_per_slice=1,
    threeD_interactive_n_init_points=1,
    threeD_patch_size=(128, 128, 128),  # This argument is only for SAMMed3D - should be different for segvol
    threeD_interactive_n_corrective_points=1,
)

pro_conf = exp_params
seed = 1
prompter = p.FiveFGPointsPer2DSlicePrompter(
    inferer,
    seed,
)
prompter.set_groundtruth(binary_gt_orig_coords)
prompt_results = prompter.predict_image(image_path=img_path)
print(prompt_results.perf)
