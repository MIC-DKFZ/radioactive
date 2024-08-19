import torchio as tio
from torchio.data.io import sitk_to_nib
import SimpleITK as sitk
import napari
import numpy as np
import nibabel as nib

from utils.class_SAMMed3D_stable import SAMMed3DInferer
from utils.prompt_3d import get_pos_clicks3D
import utils.analysis as anUt
from utils.interactivity import iterate_3d
from utils.image import read_im_gt, read_reorient_nifti

from utils.image import  read_reorient_nifti

# Obtain model, image, gt
device = 'cuda'
sammed3d_checkpoint_path = '/home/t722s/Desktop/UniversalModels/TrainedModels/sam_med3d_turbo.pth'

inferer = SAMMed3DInferer(sammed3d_checkpoint_path, device)

img_path = '/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_2img/imagesTr/BDMAP_00000001_0000.nii.gz'
gt_path = '/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_2img/labelsTr/BDMAP_00000001.nii.gz'
class_label = 3

gt_unprocessed = nib.load(gt_path).get_fdata()
gt_unprocessed = np.where(gt_unprocessed == class_label, 1, 0)

img, gt = read_im_gt(img_path, gt_path, class_label, RAS=True)

# Set image to predict on 
inferer.set_image(img_path)

# Experiment: 5 points per volume
seed = 11121
n = 5
pts_prompt = get_pos_clicks3D(gt, n, seed = seed)

#pred = inferer.predict(img, pts_prompt, cheat = True, gt = gt)
segmentation = inferer.predict(pts_prompt, store_patching=True)

anUt.compute_dice(segmentation, gt)