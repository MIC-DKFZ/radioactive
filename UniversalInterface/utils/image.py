import numpy as np
import nibabel as nib

def read_im_gt(img_path, gt_path, organ_label = None, RAS = False):
    img, gt = nib.load(img_path), nib.load(gt_path)
    img_ras, gt_ras = img, gt  # Initialize variables to hold potentially reoriented images

    # Check if gt, image are already in RAS+ 
    if nib.aff2axcodes(img.affine) != ('R', 'A', 'S'):
        img_ras = nib.as_closest_canonical(img)

    if nib.aff2axcodes(gt.affine) != ('R', 'A', 'S'):
        gt_ras = nib.as_closest_canonical(gt)

    img_data = img_ras.get_fdata().astype(np.float32)
    gt_data = gt_ras.get_fdata().astype(int)

    # Ensure organ is binary
    if organ_label is None:
        flat_data = gt_data.ravel()
        if not np.all( (flat_data == 0) | (flat_data == 1)):
            raise ValueError('Ground truth is not binary and no foreground label to subset to is specified')
        
    else:
        gt_data = (gt_data == organ_label).astype(int)

    if not RAS:
        img_data, gt_data = img_data.transpose(2,1,0), gt_data.transpose(2,1,0) # change from RAS to row-major ie xyz to zyx
    
    return(img_data, gt_data)

def read_reorient_nifti(path, RAS = False):
    img = nib.load(path)
    img_ras = img # Initialize variables to hold potentially reoriented images

    # Check if gt, image are already in RAS+ 
    if nib.aff2axcodes(img.affine) != ('R', 'A', 'S'):
        img_ras = nib.as_closest_canonical(img)

    img_data = img_ras.get_fdata()

    if not RAS:
        img_data = img_data.transpose(2,1,0) # change from RAS to row-major ie xyz to zyx
    
    return(img_data)

def get_crop_pad_params(img, crop_pad_center, target_shape): # Modified from TorchIO cropOrPad
    subject_shape = img.shape
    padding = []
    cropping = []

    for dim in range(3):
        target_dim = target_shape[dim]
        center_dim = crop_pad_center[dim]
        subject_dim = subject_shape[dim]

        center_on_index = not (center_dim % 1)
        target_even = not (target_dim % 2)

        # Approximation when the center cannot be computed exactly
        # The output will be off by half a voxel, but this is just an
        # implementation detail
        if target_even ^ center_on_index:
            center_dim -= 0.5

        begin = center_dim - target_dim / 2
        if begin >= 0:
            crop_ini = begin
            pad_ini = 0
        else:
            crop_ini = 0
            pad_ini = -begin

        end = center_dim + target_dim / 2
        if end <= subject_dim:
            crop_fin = subject_dim - end
            pad_fin = 0
        else:
            crop_fin = 0
            pad_fin = end - subject_dim

        padding.extend([pad_ini, pad_fin])
        cropping.extend([crop_ini, crop_fin])

    padding_params = np.asarray(padding, dtype=int)
    cropping_params = np.asarray(cropping, dtype=int)

    return cropping_params, padding_params  # type: ignore[return-value]

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