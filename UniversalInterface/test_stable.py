import numpy as np

from classes.SAMClass import SAMInferer
from utils.base_classes import Points
import utils.prompt as prUt
import utils.analysis as anUt
from utils.image import read_im_gt
from utils.interactivity import iterate_2d

# Obtain model
device = 'cuda'
checkpoint_path = '/home/t722s/Desktop/UniversalModels/TrainedModels/sam_vit_h_4b8939.pth'

sam_inferer = SAMInferer(checkpoint_path, 'cuda')

# Load img, gt
img_path = '/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_2img/imagesTr/BDMAP_00000001_0000.nii.gz'
gt_path = '/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_2img/labelsTr/BDMAP_00000001.nii.gz'

img, gt = read_im_gt(img_path, gt_path, 2)

# Iteratively improve starting from random points
seed = 11121
n_clicks = 5
point_prompt = prUt.get_pos_clicks2D_row_major(gt, n_clicks, seed = seed)
segmentation, low_res_logits = sam_inferer.predict(img, point_prompt, return_low_res_logits = True, use_stored_embeddings=True)
anUt.compute_dice(segmentation, gt)


initial_prompt = point_prompt
condition = 'dof'
dof_bound = 90
seed_sub = np.random.randint(10**5)
segmentation, dof, segmentations, prompts, max_fp_idxs = iterate_2d(sam_inferer, img, gt, segmentation, low_res_logits, initial_prompt, pass_prev_prompts=False, use_stored_embeddings = True,
                                                                         init_dof = 5, dof_bound = dof_bound, seed = seed_sub, detailed = True)