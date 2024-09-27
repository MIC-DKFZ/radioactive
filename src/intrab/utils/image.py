import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, ornt_transform
import torch
import torchio as tio

from intrab.prompts.prompt import PromptStep

def get_crop_pad_params_from_gt_or_prompt(img3D: np.ndarray, prompt: PromptStep | None = None, cheat: bool = False, gt: np.ndarray | None = None):
    img3D = torch.from_numpy(img3D)

    subject = tio.Subject(image=tio.ScalarImage(tensor=img3D.unsqueeze(0)))

    if cheat:
        subject.add_image(
            tio.LabelMap(
                tensor=gt.unsqueeze(0),
                affine=subject.image.affine,
            ),
            image_name="label",
        )
        crop_transform = tio.CropOrPad(mask_name="label", target_shape=(128, 128, 128))
    else:
        coords_T = prompt.coords.T
        crop_mask = torch.zeros_like(subject.image.data)
        indices = (0,) + tuple(coords_T)
        # fmt: off
        crop_mask[indices] = 1  # Include initial 0 for the additional N axis
        # fmt: on
        subject.add_image(tio.LabelMap(tensor=crop_mask, affine=subject.image.affine), image_name="crop_mask")
        crop_transform = tio.CropOrPad(mask_name="crop_mask", target_shape=(128, 128, 128))

    padding_params, cropping_params = crop_transform.compute_crop_or_pad(subject)
    # cropping_params: (x_start, x_max-(x_start+roi_size), y_start, ...)
    # padding_params: (x_left_pad, x_right_pad, y_left_pad, ...)
    if cropping_params is None:
        cropping_params = (0, 0, 0, 0, 0, 0)
    if padding_params is None:
        padding_params = (0, 0, 0, 0, 0, 0)

    return cropping_params, padding_params

def crop_im(img, cropping_params): # Modified from TorchIO cropOrPad
        low = cropping_params[::2]
        high = cropping_params[1::2]
        index_ini = low
        index_fin = np.array(img.shape) - high 
        i0, j0, k0 = index_ini
        i1, j1, k1 = index_fin
        image_cropped = img[i0:i1, j0:j1, k0:k1]

        return(image_cropped)

def pad_im(img, padding_params): # Modified from TorchIO cropOrPad
    paddings = padding_params[:2], padding_params[2:4], padding_params[4:]
    image_padded = np.pad(img, paddings, mode = 'constant', constant_values = 0)  

    return(image_padded)

def invert_crop_or_pad(mask, cropping_params, padding_params):
    if padding_params is not None:
        mask = crop_im(mask, padding_params)
    if cropping_params is not None:
        mask = pad_im(mask, cropping_params)
    return(mask)