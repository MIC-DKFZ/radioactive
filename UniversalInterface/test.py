import numpy as np
import nibabel as nib

from utils.class_SAM import SAMInferer
from utils.base_classes import Points, Prompt
import utils.prompt as prUt
import utils.analysis as anUt
from utils.image import read_im_gt
from utils.interactivity import iterate_2d

# Obtain model
device = 'cuda'
checkpoint_path = '/home/t722s/Desktop/UniversalModels/TrainedModels/sam_vit_h_4b8939.pth'

sam_inferer = SAMInferer(checkpoint_path, device)

# Load img, gt
img_path = '/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_2img/imagesTr/BDMAP_00000001_0000.nii.gz'
gt_path = '/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_2img/labelsTr/BDMAP_00000001.nii.gz'
class_label = 2 

gt_unprocessed = nib.load(gt_path).get_fdata()
gt_unprocessed = np.where(gt_unprocessed == class_label, 1, 0)

img, gt = read_im_gt(img_path, gt_path, class_label)

# Iteratively improve starting from random points
seed = 11121
n_clicks = 5
point_prompt = prUt.get_pos_clicks2D_row_major(gt, n_clicks, seed = seed)
segmentation, low_res_logits = sam_inferer.predict(img_path, point_prompt, return_low_res_logits = True, use_stored_embeddings=True, transform = False)
anUt.compute_dice(segmentation, gt)


initial_prompt = point_prompt
condition = 'dof'
dof_bound = 90
seed_sub = np.random.randint(10**5)
dice_scores, dofs, segmentations, prompts = iterate_2d(sam_inferer, img_path, gt, segmentation, low_res_logits, initial_prompt, pass_prev_prompts=True, use_stored_embeddings = True,
                                                                         init_dof = 5, dof_bound = dof_bound, seed = seed_sub, detailed = True)

print(dice_scores)
pass