import numpy as np
import nibabel as nib
from intrab.utils.nnunet.resample_torch import resample_torch_fornnunet
from intrab.utils.nnunet.default_resampling import compute_new_shape
from intrab.utils.analysis import compute_dice


# Load in images for resampling
gt_path = '/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_2img/labelsTr/BDMAP_00000001.nii.gz'
img_path = '/home/t722s/Desktop/Datasets/Dataset350_AbdomenAtlasJHU_2img/imagesTr/BDMAP_00000001_0000.nii.gz'
seg_path = '/home/t722s/Desktop/ExperimentResults/metric_testing/bounding_boxes/kidney_left/BDMAP_00000001.nii.gz'

gt = nib.load(gt_path).get_fdata()
gt = np.where(gt == 3, 1, 0)
img = nib.load(img_path).get_fdata()
seg = nib.load(seg_path).get_fdata()


from typing import Tuple


def get_current_spacing_from_affine(affine: np.ndarray):
    return np.abs(affine.diagonal()[:-1])

def resample(seg:np.ndarray, current_spacing: Tuple|np.ndarray, new_spacing: Tuple|np.ndarray = (1.5, 1.5, 1.5), is_seg:bool = True):
    # Find new shape to change to
    new_shape = compute_new_shape(seg.shape, current_spacing, new_spacing)
    # Change gt to cxyz
    seg = seg[None]
    # Perform resampling
    seg_resampled = resample_torch_fornnunet(
        seg,
        new_shape,
        current_spacing,
        new_spacing, 
        is_seg = is_seg,
        device = 'cuda', memefficient_seg_resampling=True
    )

    seg_resampled = seg_resampled[0]

    return(seg_resampled)

new_spacing = (1.5,1.5,1.5)

gt_nib = nib.load(gt_path)
gt_current_spacing =  get_current_spacing_from_affine(gt_nib.affine)# ignore affine component of diagonal

gt = gt_nib.get_fdata()
gt = np.where(gt == 3, 1, 0)
gt_resampled = resample(gt, gt_current_spacing, is_seg = True)

seg_nib = nib.load(seg_path)
seg_current_spacing =  get_current_spacing_from_affine(seg_nib.affine)# ignore affine component of diagonal

seg = seg_nib.get_fdata()
seg_resampled = resample(seg, seg_current_spacing, is_seg = True)

print(compute_dice(seg, gt), compute_dice(seg_resampled, gt_resampled))