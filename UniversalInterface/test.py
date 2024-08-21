from utils.prompt_3d import get_bbox3d, get_pos_clicks3D
from utils.analysis import compute_dice
import numpy as np
import nibabel as nib
from importlib import reload
import utils.class_segvol as c

# Obtain model, image, gt
device = 'cuda' # In this case, redundant; segvol requires cuda
checkpoint_path = "/home/t722s/Desktop/UniversalModels/TrainedModels/SegVol_v1.pth"

inferer = c.SegVolInferer(checkpoint_path)
#inferer = SegVolInferer(checkpoint)

img_path = "/home/t722s/Desktop/Datasets/segvolTest/Case_image_00001_0000.nii.gz"
gt_path = "/home/t722s/Desktop/Datasets/segvolTest/Case_label_00001.nii.gz" 
class_label = 1

gt = nib.load(gt_path).get_fdata()
gt = np.where(gt == class_label, 1, 0)
inferer.set_image(img_path)

# Experiment: 5 points per volume
seed = 11121
inferer.set_image(img_path)

prompt = get_pos_clicks3D(gt, 5, seed)
print(prompt.coords)
#prompt.coords = prompt.coords[:,::-1]
segmentation = inferer.predict(prompt).get_fdata()

print(compute_dice(segmentation, gt))
