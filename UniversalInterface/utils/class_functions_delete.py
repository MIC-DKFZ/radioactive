import matplotlib.pyplot as plt
import torch
import numpy as np
from skimage.morphology import dilation, ball, disk
from .base_classes import Points, Boxes2d


def binarize(gt, organ, metadata): # TODO: Raise errors for metadata not having labels, labels in the wrong k,v order, and organ not being present
    organ_label = int(metadata['labels'][organ])
    gt_binary = torch.where(gt == organ_label, 1, torch.zeros_like(gt))
    return(gt_binary)

def get_crop_pad_center_from_points(points):
    bbox_min = points.value['coords'].T.min(axis = 1) # Get an array of two points: the minimal and maximal vertices of the minimal cube parallel to the axes bounding the points
    bbox_max = points.value['coords'].T.max(axis = 1) + 1 # Add 1 since we'll be using this for indexing # TESTING: Remove 'self's here
    point_center = np.mean((bbox_min, bbox_max), axis = 0)  

    return(point_center)

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

def crop_pad_coords(coords, cropping_params, padding_params):
    axis_add, axis_sub = padding_params[::2], cropping_params[::2] 
    coords = coords + axis_add - axis_sub # same as value[:,i] = value[:,i] + axis_add[i] - axis_sub[i] iterating over i
    return(coords)

def invert_crop_or_pad(mask, cropping_params, padding_params):
    if padding_params is not None:
        mask = crop_im(mask, padding_params)
    if cropping_params is not None:
        mask = pad_im(mask, cropping_params)
    return(mask)

def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum

def get_pos_clicks3D(gt, n_clicks):
    volume_fg = np.where(gt==1) # Get foreground indices (formatted as triple of arrays)
    volume_fg = np.array(volume_fg).T # Reformat to numpy array of shape n_fg_voxels x 3

    n_fg_voxels = len(volume_fg)

    # Error testing
    if n_fg_voxels == 0:
        raise RuntimeError(f'No foreground voxels found! Check that the supplied label is a binary segmentation mask with foreground coded as 1')

    if n_fg_voxels < n_clicks:
        raise RuntimeError(f'More foreground points were requested than the number of foreground voxels in the volume')

    point_indices = np.random.choice(n_fg_voxels, size = n_clicks, replace = False)
    pos_coords = volume_fg[point_indices]  
    pos_coords = Points({'coords': pos_coords, 'labels': [1]*len(pos_coords)})
    return(pos_coords)

def get_neg_clicks_3D(gt, n_clicks, border_distance = 10): # Warning: dilation function is VERY slow! ~ 13 seconds on my machine
    struct_element = ball(border_distance)
    volume_dilated = dilation(gt, struct_element)

    border_region = volume_dilated - gt

    volume_border = np.where(border_region)
    volume_border = np.array(volume_border).T
    n_border_voxels = volume_border.shape[0]

    if n_border_voxels < n_clicks:
        raise RuntimeError(f'More background points were requested than the number of border voxels in the volume ({n_clicks} vs {n_border_voxels})')

    point_indices = np.random.choice(n_border_voxels, size = n_clicks, replace = False)
    neg_coords = volume_border[point_indices]  # change from triple of arrays format to list of triples format # change from triple of arrays format to list of triples format

    neg_coords = Points({'coords': neg_coords, 'labels': [0]*len(neg_coords)})
    return(neg_coords)

def get_pos_clicks2D(gt, n_clicks):
    volume_fg = np.where(gt==1) # Get foreground indices (formatted as triple of arrays)
    volume_fg = np.array(volume_fg).T # Reformat to numpy array of shape n_fg_voxels x 3
    
    fg_slices = np.unique(volume_fg[:,2]) # Obtain superior axis slices which have foreground before reformating indices

    pos_coords = np.empty(shape = (0,3), dtype = int)
    warning_zs = [] # track slices without enough foreground/border, if any should exist

    for slice_index in fg_slices:
        ## Foreground points
        slice = gt[:,:,slice_index]
        slice_fg = np.where(slice)
        slice_fg = np.array(slice_fg).T

        n_fg_pixels = len(slice_fg)
        if n_fg_pixels >= n_clicks:
            point_indices = np.random.choice(n_fg_pixels, size = n_clicks, replace = False)
        else:
            # In this case, take all foreground pixels and then obtain some duplicate points by sampling with replacement additionally
            warning_zs.append(f'z = {slice_index}, n foreground = n_fg_pixels')
            point_indices = np.concatenate([np.arange(n_fg_pixels),
                                        np.random.choice(n_fg_pixels, size = n_clicks-n_fg_pixels, replace = True)])
            
        pos_clicks_slice = slice_fg[point_indices]
        z_col = np.full((n_clicks,1), slice_index) # create z column to add
        pos_clicks_slice = np.hstack([pos_clicks_slice, z_col])
        pos_coords = np.vstack([pos_coords, pos_clicks_slice])

    pos_coords = Points({'coords': pos_coords, 'labels': [1]*len(pos_coords)})
    return(pos_coords)

def get_bbox3d(mask_volume: np.ndarray):
    """Return 6 coordinates of a 3D bounding box from a given mask.

    Taken from `this SO question <https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array>`_.

    Args:
        mask_volume: 3D NumPy array.
    """  # noqa: B950
    i_any = np.any(mask_volume, axis=(1, 2))
    j_any = np.any(mask_volume, axis=(0, 2))
    k_any = np.any(mask_volume, axis=(0, 1))
    i_min, i_max = np.where(i_any)[0][[0, -1]]
    j_min, j_max = np.where(j_any)[0][[0, -1]]
    k_min, k_max = np.where(k_any)[0][[0, -1]]
    bb_min = np.array([i_min, j_min, k_min])
    bb_max = np.array([i_max, j_max, k_max]) + 1
    return bb_min, bb_max

def get_bbox2d(mask):
    i_any = np.any(mask, axis=1)
    j_any = np.any(mask, axis=(0))
    i_min, i_max = np.where(i_any)[0][[0, -1]]
    j_min, j_max = np.where(j_any)[0][[0, -1]]
    bb_min = np.array([i_min, j_min])
    bb_max = np.array([i_max, j_max]) + 1

    coord_list = np.concatenate([bb_min, bb_max]) # SAM wants row-major y0, x0, y1, x1
    return coord_list

def get_minimal_boxes(gt, delta_x, delta_y):
    '''
    Get bounding boxes of the foreground per slice. delta_x, delta_y enlargen the box
    in the respective dimensions
    '''
    slices_to_infer = np.where(np.any(gt, axis=(0,1)))[0] # index 0 since a tuple of length 1 is returned
    box_dict = {slice_idx: get_bbox2d(gt[:,:,slice_idx].T) for slice_idx in slices_to_infer} # Transpose to get coords in row-major format
    box_dict = {slice_idx: box + np.array([-delta_x, -delta_y, delta_x, delta_y]) for slice_idx, box in box_dict.items()}
    return(Boxes2d(box_dict))

def get_3d_box_for_2d(gt, delta_x, delta_y):
    '''
    Finds 3d bounding box over the volume and returns it in a 2d prompt
    '''
    box_3d = get_bbox3d(gt.transpose(2,1,0)) # Permute to get coords in row major format
    box_2d = np.concatenate([box_3d[0][-2:], box_3d[1][-2:]])
    box_2d = box_2d + np.array([-delta_x, -delta_y, delta_x, delta_y])
    box_dict = {slice_idx: box_2d for slice_idx in range(box_3d[0][2], box_3d[1][2])}
    return(Boxes2d(box_dict))

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

def show_seg_box(slice_idx, img, gt, segmentation, box_prompt):
    img_2d = img[..., slice_idx]
    gt_2d = gt[..., slice_idx]
    seg_2d = segmentation[..., slice_idx]
    box = box_prompt.value[slice_idx]

    img_2d = (img_2d-img_2d.min())/(img_2d.max()-img_2d.min())

    # visualize results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_2d, cmap = 'gray')
    show_box(box, ax[0])
    ax[0].set_title("Input Image and Bounding Box")
    ax[1].imshow(gt_2d, cmap = 'gray')
    show_mask(seg_2d, ax[1])
    show_box(box, ax[1])
    ax[1].set_title("MedSAM Segmentation")
    plt.show()
    slice_dice = compute_dice(seg_2d, gt_2d)
    return(slice_dice)

def show_seg(slice_idx, img, gt, segmentation, pts_prompt = None, box_prompt = None):
    img_2d = img[..., slice_idx]
    gt_2d = gt[..., slice_idx]
    seg_2d = segmentation[..., slice_idx]

    img_2d = (img_2d-img_2d.min())/(img_2d.max()-img_2d.min())

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_2d, cmap = 'gray')
    ax[1].imshow(gt_2d, cmap = 'gray')
    show_mask(seg_2d, ax[1])
    ax[1].set_title("MedSAM Segmentation")
    ax[0].set_title("Input Image and Bounding Box")

    if pts_prompt is not None:
        coords, _ = pts_prompt.value.values()
        slice_inds = coords[:,2] == slice_idx
        slice_coords = coords[slice_inds,:-1].T
        ax[0].plot(slice_coords[1], slice_coords[0], 'ro')
        ax[1].plot(slice_coords[1], slice_coords[0], 'ro')
        
    if box_prompt is not None:
        box = box_prompt.value[slice_idx]
        show_box(box, ax[0])
        show_box(box, ax[0])
    plt.show()
    return(compute_dice(seg_2d, gt_2d))