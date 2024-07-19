import torchio as tio
from torchio.data.io import sitk_to_nib
import SimpleITK as sitk
import napari

from classes.SAMMed3DClass import SAMMed3DInferer
from utils.prompt import get_pos_clicks3D
import utils.analysis as anUt

from utils.image import read_im_gt


def get_img_gt_sammed3d(img_path, gt_path):    
    infer_transform = [
        tio.ToCanonical(),
    ]
    transform = tio.Compose(infer_transform)

    sitk_image = sitk.ReadImage(img_path)
    sitk_label = sitk.ReadImage(gt_path)

    if sitk_image.GetOrigin() != sitk_label.GetOrigin():
        sitk_image.SetOrigin(sitk_label.GetOrigin())
    if sitk_image.GetDirection() != sitk_label.GetDirection():
        sitk_image.SetDirection(sitk_label.GetDirection())

    sitk_image_arr, _ = sitk_to_nib(sitk_image)
    sitk_label_arr, _ = sitk_to_nib(sitk_label)

    subject = tio.Subject(
        image = tio.ScalarImage(tensor=sitk_image_arr),
        label = tio.LabelMap(tensor=sitk_label_arr),
    )

    if transform:
        subject = transform(subject)

    return subject.image.data.clone().detach().squeeze().numpy(), subject.label.data.clone().detach().squeeze().numpy()

# Obtain model, image, gt
device = 'cuda'
sammed3d_checkpoint_path = '/home/t722s/Desktop/UniversalModels/TrainedModels/sam_med3d_turbo.pth'

inferer = SAMMed3DInferer(sammed3d_checkpoint_path, device)

img_path = '/home/t722s/Desktop/Datasets/preprocessed/spleen/AbdomenAtlasJHU_2img/imagesTr/BDMAP_00000001.nii.gz'
gt_path = '/home/t722s/Desktop/Datasets/preprocessed/spleen/AbdomenAtlasJHU_2img/labelsTr/BDMAP_00000001.nii.gz'
#img, gt = get_img_gt_sammed3d(img_path, gt_path)
img, gt = read_im_gt(img_path, gt_path)


# Experiment: 5 points per volume
seed = 11121
n = 5
pts_prompt = get_pos_clicks3D(gt, n, seed = seed)

#pred = inferer.predict(img, pts_prompt, cheat = True, gt = gt)
segmentation = inferer.predict(img, pts_prompt)

print(anUt.compute_dice(segmentation, gt))