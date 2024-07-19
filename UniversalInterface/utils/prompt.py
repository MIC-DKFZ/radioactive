import numpy as np
from copy import deepcopy
from tqdm import tqdm
from skimage.morphology import dilation, ball
from skimage.measure import label
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import warnings
import torch

from .base_classes import Points, Boxes
from .analysis import compute_dice

def get_crop_pad_center_from_points(points):
    bbox_min = points.coords.T.min(axis = 1) # Get an array of two points: the minimal and maximal vertices of the minimal cube parallel to the axes bounding the points
    bbox_max = points.coords.T.max(axis = 1) + 1 # Add 1 since we'll be using this for indexing 
    point_center = np.mean((bbox_min, bbox_max), axis = 0)  

    return(point_center)

def crop_pad_coords(coords, cropping_params, padding_params):
    axis_add, axis_sub = padding_params[::2], cropping_params[::2] 
    coords = coords + axis_add - axis_sub # same as value[:,i] = value[:,i] + axis_add[i] - axis_sub[i] iterating over i
    return(coords)

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

    neg_coords = Points(coords = neg_coords, labels =  [0]*len(neg_coords))
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

    pos_coords = Points(coords = pos_coords, labels =  [1]*len(pos_coords))
    return(pos_coords)

def get_pos_clicks2D_row_major(gt, n_clicks, seed = None):

    if seed is not None:
        np.random.seed(seed)
    volume_fg = np.where(gt==1) # Get foreground indices (formatted as triple of arrays)
    volume_fg = np.array(volume_fg).T # Reformat to numpy array of shape n_fg_voxels x 3
    
    fg_slices = np.unique(volume_fg[:,0]) # Obtain superior axis slices which have foreground before reformating indices

    pos_coords = np.empty(shape = (0,3), dtype = int)
    warning_zs = [] # track slices without enough foreground/border, if any should exist

    for slice_index in fg_slices:
        ## Foreground points
        slice = gt[slice_index,:,:]
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
        pos_clicks_slice = np.hstack([z_col, pos_clicks_slice])
        pos_coords = np.vstack([pos_coords, pos_clicks_slice])

    pos_coords = Points(coords = pos_coords, labels =  np.array([1]*len(pos_coords)))
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

def get_bbox3d_sliced(mask_volume: np.ndarray):
    bbox3d = get_bbox3d(mask_volume)

    slices_to_infer = np.arange(bbox3d[0][0], bbox3d[1][0]) # gt is in ZYX format, so index 0 are the axial slices
    box_dict = {slice_idx: np.array((bbox3d[0][2], bbox3d[0][1], bbox3d[1][2], bbox3d[1][1])) for slice_idx in slices_to_infer} # reverse order to get xyxy
    return(Boxes(box_dict))

def get_bbox2d(mask):
    i_any = np.any(mask, axis=1)
    j_any = np.any(mask, axis=0)
    i_min, i_max = np.where(i_any)[0][[0, -1]]
    j_min, j_max = np.where(j_any)[0][[0, -1]]
    bb_min = np.array([i_min, j_min])
    bb_max = np.array([i_max, j_max]) + 1

    coord_list = np.concatenate([bb_min, bb_max]) # SAM wants row-major x0y0x1y1
    return coord_list

def get_minimal_boxes(gt, delta_x, delta_y):
    '''
    Get bounding boxes of the foreground per slice. delta_x, delta_y enlargen the box
    in the respective dimensions
    '''
    slices_to_infer = np.where(np.any(gt, axis=(0,1)))[0] # index 0 since a tuple of length 1 is returned
    box_dict = {slice_idx: get_bbox2d(gt[:,:,slice_idx].T) for slice_idx in slices_to_infer} # Transpose to get coords in row-major format
    box_dict = {slice_idx: box + np.array([-delta_x, -delta_y, delta_x, delta_y]) for slice_idx, box in box_dict.items()}
    return(Boxes(box_dict))

def get_bbox2d_row_major(mask):
    '''
    Mask must be in row major roder
    '''
    i_any = np.any(mask, axis=1)
    j_any = np.any(mask, axis=0)
    i_min, i_max = np.where(i_any)[0][[0, -1]]
    j_min, j_max = np.where(j_any)[0][[0, -1]]
    bb_min = np.array([j_min, i_min])
    bb_max = np.array([j_max, i_max]) + 1

    coord_list = np.concatenate([bb_min, bb_max]) 
    return coord_list

def get_minimal_boxes_row_major(gt, delta_x=0, delta_y=0):
    '''
    gt must be in row-major ZYX order.
    Get bounding boxes of the foreground per slice. delta_x, delta_y enlargen the box
    in the respective dimensions
    '''
    slices_to_infer = np.where(np.any(gt, axis=(1,2)))[0] # index 0 since a tuple of length 1 is returned
    box_dict = {slice_idx: get_bbox2d_row_major(gt[slice_idx,:,:]) for slice_idx in slices_to_infer} 
    box_dict = {slice_idx: box + np.array([-delta_x, -delta_y, delta_x, delta_y]) for slice_idx, box in box_dict.items()}
    return(Boxes(box_dict))

# Have to rework for new row-major order
# def get_3d_box_for_2d(gt, delta_x, delta_y):
#     '''
#     Finds 3d bounding box over the volume and returns it in a 2d prompt
#     '''
#     box_3d = get_bbox3d(gt.transpose(2,1,0)) # Permute to get coords in row major format
#     box_2d = np.concatenate([box_3d[0][-2:], box_3d[1][-2:]])
#     box_2d = box_2d + np.array([-delta_x, -delta_y, delta_x, delta_y])
#     box_dict = {slice_idx: box_2d for slice_idx in range(box_3d[0][2], box_3d[1][2])}
#     return(Boxes(box_dict))

def get_nearest_fg_point(point, binary_mask):
    """
    Find the nearest foreground coordinate (value 1) to the given 2D point in a binary mask.

    Parameters:
    point (tuple): The (y, x) coordinates of the point.
    binary_mask (numpy.ndarray): The 2D binary mask.

    Returns:
    tuple: The (y, x) coordinates of the nearest foreground point.
    """
    # Get the indices of the foreground pixels
    fg_coords = np.column_stack(np.where(binary_mask == 1))
    
    # Create a KD-tree from the foreground coordinates
    tree = cKDTree(fg_coords)
    
    # Query the nearest single neighbor
    distance, index = tree.query([point], k=1)
    nearest_fg_point = fg_coords[index[0]]

    return tuple(nearest_fg_point)
  
def get_largest_CC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC.astype(int)

def get_fg_points_from_cc_centers(gt, n):
    def get_bbox(mask): # Bbox function only used in this function
        i_any = np.any(mask, axis=1)
        j_any = np.any(mask, axis=0)
        i_min, i_max = np.where(i_any)[0][[0, -1]]
        j_min, j_max = np.where(j_any)[0][[0, -1]]
        bb_min = np.array([i_min, j_min])
        bb_max = np.array([i_max, j_max])

        return bb_min, bb_max
        
    '''
    Takes n equally spaced slices starting at z_min and ending at z_max, where z_min is the lowest transverse slice of gt containing fg, and z_max similarly with highest, 
    finds the largest connected component of fg, takes the center of its bounding box and takes the nearest fg point. Simulates a clinician clicking in the 'center of the main mass of the roi' per slice
    '''
    z_indices = np.where(np.any(gt, axis = (1,2)))[0]
    min_z, max_z = np.min(z_indices), np.max(z_indices)
    selected_slices = np.linspace(min_z, max_z, num=n, dtype=int)

    corrected_points = np.empty([n, 3], dtype = int)
    for i, z in enumerate(selected_slices): #selected_slices:
        largest_cc = get_largest_CC(gt[z])
        bbox_min, bbox_max = get_bbox(largest_cc)

        slice_fg_center = np.vstack([bbox_min, bbox_max]).mean(axis=0).round()
        
        nearest_fg_point = get_nearest_fg_point(slice_fg_center, largest_cc)
        corrected_points[i] = np.concatenate([[z], nearest_fg_point])    

    return(corrected_points)

def interpolate_points(points, kind='linear'):
    """
    Interpolate points in 3D space using linear or cubic spline interpolation.

    Parameters:
    points (numpy.ndarray): An array of shape (n, 3) where each row represents (z, y, x).
    kind (str): Type of interpolation, either 'linear' or 'cubic'.

    Returns:
    numpy.ndarray: Array of interpolated points at each integer z-coordinate within the range of input z-coordinates.
    """
    # Ensure points are sorted by z-coordinate
    points = points[points[:, 0].argsort()]

    # Separate z, y, and x coordinates
    z, y, x = points[:, 0], points[:, 1], points[:, 2]

    # Create interpolation functions for y and x as functions of z
    y_interp = interp1d(z, y, kind=kind)
    x_interp = interp1d(z, x, kind=kind)

    # Create an array of z values for which we want to interpolate y and x
    z_new = np.arange(z.min(), z.max() + 1)
    y_new = y_interp(z_new)
    x_new = x_interp(z_new)

    # Stack the new z, y, and x coordinates vertically and return
    return np.column_stack((z_new, y_new, x_new)).round()

def point_interpolation(simulated_clicks):
    coords = interpolate_points(simulated_clicks, kind = 'linear').astype(int)
    point_prompt = Points(coords = coords, labels =  [1]*len(coords))
    return(point_prompt)

def get_fg_points_from_slice(slice, n_clicks, seed = None):
    if seed:
        np.random.seed(seed)
    slice_fg = np.where(slice)
    slice_fg = np.array(slice_fg).T

    n_fg_pixels = len(slice_fg)
    if n_fg_pixels >= n_clicks:
        point_indices = np.random.choice(n_fg_pixels, size = n_clicks, replace = False)
    else:
        # In this case, take all foreground pixels and then obtain some duplicate points by sampling with replacement additionally
        point_indices = np.concatenate([np.arange(n_fg_pixels),
                                    np.random.choice(n_fg_pixels, size = n_clicks-n_fg_pixels, replace = True)])
        
    pos_clicks_slice = slice_fg[point_indices]
    return(pos_clicks_slice)

def get_seed_boxes(gt, n):
    z_indices = np.where(np.any(gt, axis = (1,2)))[0]
    min_z, max_z = np.min(z_indices), np.max(z_indices)
    selected_slices = np.linspace(min_z, max_z, num=n, dtype=int)

    bbox_dict = {slice_idx: get_bbox2d_row_major(gt[slice_idx]) for slice_idx in selected_slices}

    return(bbox_dict)

def box_interpolation(seed_boxes):

        
    '''
    Takes n equally spaced slices starting at z_min and ending at z_max, where z_min is the lowest transverse slice of gt containing fg, and z_max similarly with highest, 
    finds the largest connected component of fg, takes the center of its bounding box and takes the nearest fg point. Simulates a clinician clicking in the 'center of the main mass of the roi' per slice
    '''

    bbox_mins = np.array([(slice_idx, bbox[0], bbox[1]) for slice_idx, bbox in seed_boxes.items()])
    bbox_mins_interpolated = interpolate_points(bbox_mins)

    bbox_maxs = np.array([(slice_idx, bbox[2], bbox[3]) for slice_idx, bbox in seed_boxes.items()])
    bbox_maxs_interpolated = interpolate_points(bbox_maxs)

    bbox_np_interpolated = np.concatenate((bbox_mins_interpolated, bbox_maxs_interpolated[:, 1:]), axis = 1).astype(int)

    bbox_dict_interpolated = {row[0]: row[1:] for row in bbox_np_interpolated }
    box_prompt = Boxes(bbox_dict_interpolated)

    return(box_prompt)

def get_seed_point(gt, n_clicks, seed):
    slices_to_infer = np.where(np.any(gt, axis=(1,2)))[0]
    middle_idx = np.median(slices_to_infer).astype(int)

    pos_clicks_slice = get_fg_points_from_slice(gt[middle_idx], n_clicks, seed)

    ## Put coords in 3d context
    z_col = np.full((n_clicks,1), middle_idx) 
    pos_coords = np.hstack([z_col, pos_clicks_slice])
    pts_prompt = Points(coords = pos_coords, labels = [1]*n_clicks)

    return(pts_prompt)

def point_propagation(inferer, img, seed_prompt, slices_to_infer, seed = None, n_clicks=5, verbose = True, return_low_res_logits = False):
    if seed:
        np.random.seed(seed)
    verbose_state = inferer.verbose # Make inferer not verbose for this experiment
    inferer.verbose = False

    seed_coords = [seed_prompt.coords] # keep track of all points
    
    # Initialise segmentation to store total result
    segmentation = np.zeros_like(img).astype(np.uint8)
    if return_low_res_logits:
        low_res_logits = {}

    pts_prompt = deepcopy(seed_prompt)
    middle_idx = np.median(slices_to_infer).astype(int)

    # Infer middle slice
    if return_low_res_logits:
        slice_seg, slice_low_res_logits = inferer.predict(img, pts_prompt, return_low_res_logits = True)
        low_res_logits[middle_idx] = slice_low_res_logits[middle_idx]
    else:
        slice_seg = inferer.predict(img, pts_prompt)
    segmentation[middle_idx] = slice_seg[middle_idx]

    # Downwards branch
    ## Modify seed prompt to exist one axial slice down
    downwards_coords = []
    z_col = np.full((len(seed_prompt.coords),1), middle_idx-1) 
    pos_coords = np.hstack([z_col, seed_prompt.coords[:,1:]])
    downwards_coords.append(pos_coords)
    pts_prompt = Points(coords = pos_coords, labels =[1]*len(seed_prompt.coords))


    downwards_iter = range(middle_idx-1, slices_to_infer.min()-1, -1)
    if verbose:
        downwards_iter = tqdm(downwards_iter, desc = 'Propagating down')

    for slice_idx in downwards_iter:    
        if return_low_res_logits:
            slice_seg, slice_low_res_logits = inferer.predict(img, pts_prompt, return_low_res_logits = True)
            low_res_logits[slice_idx] = slice_low_res_logits[slice_idx]
        else:
            slice_seg = inferer.predict(img, pts_prompt)
        segmentation[slice_idx] = slice_seg[slice_idx]

        if np.all(segmentation[slice_idx] == 0): # Terminate if no fg generated
            warnings.warn('\nTerminate early: no fg generated')
            break

        # Update prompt
        pos_clicks_slice = get_fg_points_from_slice(slice_seg[slice_idx], n_clicks)

        # Put coords in 3d context
        z_col = np.full((n_clicks,1), slice_idx-1) # create z column to add. Note slice_idx-1: these are the prompts for the next slice down
        pos_coords = np.hstack([z_col, pos_clicks_slice])
        downwards_coords.append(pos_coords)
        pts_prompt = Points(coords = pos_coords, labels = [1]*n_clicks)

    # Upward branch
    ## Modify seed prompt to exist one axial slice up
    upward_coords = []
    z_col = np.full((len(seed_prompt.coords),1), middle_idx+1) 
    pos_coords = np.hstack([z_col, seed_prompt.coords[:,1:]])
    upward_coords.append(pos_coords)
    pts_prompt = Points(coords = pos_coords, labels =  [1]*len(seed_prompt.coords))


    upwards_iter = range(middle_idx+1, slices_to_infer.max()+1)
    if verbose:
        upwards_iter = tqdm(upwards_iter, desc = 'Propagating up')

    for slice_idx in upwards_iter:
        if return_low_res_logits:
            slice_seg, slice_low_res_logits = inferer.predict(img, pts_prompt, return_low_res_logits = True)
            low_res_logits[slice_idx] = slice_low_res_logits[slice_idx]
        else:
            slice_seg = inferer.predict(img, pts_prompt)

        segmentation[slice_idx] = slice_seg[slice_idx]

        if np.all(segmentation[slice_idx] == 0): # Terminate if no fg generated
            warnings.warn('\nTerminate early: no fg generated')
            break

        # Update prompt
        pos_clicks_slice = get_fg_points_from_slice(slice_seg[slice_idx], n_clicks)

        # Put coords in 3d context
        z_col = np.full((n_clicks,1), slice_idx+1) # create z column to add. Note slice_idx+1: these are the prompts for the next slice up
        pos_coords = np.hstack([z_col, pos_clicks_slice])
        upward_coords.append(pos_coords)
        pts_prompt = Points(coords = pos_coords, labels =  [1]*n_clicks)

    inferer.verbose = verbose_state # Return inferer verbosity to initial state

    coords = np.concatenate(downwards_coords[::-1] + seed_coords + upward_coords, axis = 0)
    is_in_slices_inferred = np.isin(coords[:,0], slices_to_infer)
    coords = coords[is_in_slices_inferred] # Removes extraneous prompts on bottom_slice-1 and top_slice+1 that weren't used.
    prompt = Points(coords = coords, labels =  [1]*len(coords))

    if return_low_res_logits:
        return segmentation, low_res_logits, prompt
    else:
        return segmentation, prompt


def get_seed_box(gt):
    slices_to_infer = np.where(np.any(gt, axis=(1,2)))[0]
    middle_idx = np.median(slices_to_infer).astype(int)

    bbox_slice = get_bbox2d_row_major(gt[middle_idx])
    box_dict = {middle_idx: bbox_slice}

    return(Boxes(box_dict))

def box_propagation(inferer, img, seed_box, slices_to_infer, verbose = True, return_low_res_logits = False):
    verbose_state = inferer.verbose # Make inferer not verbose for this experiment
    inferer.verbose = False

    # Initialise segmentation to store total result
    segmentation = np.zeros_like(img).astype(np.uint8)
    if return_low_res_logits:
        low_res_logits = {}

    box_prompt = deepcopy(seed_box)
    all_boxes = seed_box.value # Update throughout the loops to keep track of all box prompts
    middle_idx = np.median(slices_to_infer).astype(int)

    # Infer middle slice
    if return_low_res_logits:
        slice_seg, slice_low_res_logits = inferer.predict(img, box_prompt, return_low_res_logits = True)
        low_res_logits[middle_idx] = slice_low_res_logits[middle_idx]
    else:
        slice_seg = inferer.predict(img, box_prompt)
    segmentation[middle_idx] = slice_seg[middle_idx]


    # Downwards branch
    ## Modify seed prompt to exist one axial slice down
    all_boxes[middle_idx-1] = all_boxes[middle_idx] 
    box_prompt = Boxes({k-1:v for k,v in seed_box.value.items()})

    downwards_iter = range(middle_idx-1, slices_to_infer.min()-1, -1)
    if verbose:
        downwards_iter = tqdm(downwards_iter, desc = 'Propagating down')

    for slice_idx in downwards_iter:
        if return_low_res_logits:
            slice_seg, slice_low_res_logits = inferer.predict(img, box_prompt, return_low_res_logits = True)
            low_res_logits[slice_idx] = slice_low_res_logits[slice_idx]
        else:
            slice_seg = inferer.predict(img, box_prompt)

        segmentation[slice_idx] = slice_seg[slice_idx]

        if np.all(segmentation[slice_idx] == 0): # Terminate if no fg generated
            warnings.warn('\nTerminate early: no fg generated')
            break

        # Update prompt
        bbox_slice = get_bbox2d_row_major(segmentation[slice_idx])
        all_boxes[slice_idx-1] = bbox_slice
        box_prompt = Boxes({slice_idx-1: bbox_slice}) # Notice the -1: this is the prompt for one slice down

    # Upward branch
    ## Modify seed prompt to exist one axial slice up
    all_boxes[middle_idx+1] = all_boxes[middle_idx]
    box_prompt = Boxes({k+1:v for k,v in seed_box.value.items()})

    upwards_iter = range(middle_idx+1, slices_to_infer.max()+1)
    if verbose:
        upwards_iter = tqdm(upwards_iter, desc = 'Propagating up')

    for slice_idx in upwards_iter:
        if return_low_res_logits:
            slice_seg, slice_low_res_logits = inferer.predict(img, box_prompt, return_low_res_logits = True)
            low_res_logits[slice_idx] = slice_low_res_logits[slice_idx]
        else:
            slice_seg = inferer.predict(img, box_prompt)

        segmentation[slice_idx] = slice_seg[slice_idx]

        if np.all(segmentation[slice_idx] == 0): # Terminate if no fg generated
            warnings.warn('\nTerminate early: no fg generated')
            break

        # Update prompt
        bbox_slice = get_bbox2d_row_major(segmentation[slice_idx])
        all_boxes[slice_idx+1] = bbox_slice
        box_prompt = Boxes({slice_idx+1: bbox_slice}) # Notice the +1: this is the prompt for one slice up
    
    all_boxes = {k:all_boxes[k] for k in slices_to_infer} # Removes top and bottom box - they weren't used

    all_boxes = Boxes(all_boxes)
    inferer.verbose = verbose_state # Return inferer verbosity to initial state
    return segmentation, all_boxes

def line_interpolation(gt, n_slices, interpolation = 'linear'):
    simulated_clicks = get_fg_points_from_cc_centers(gt, n_slices)
    coords = interpolate_points(simulated_clicks, kind = interpolation).astype(int)
    point_prompt = Points(coords = coords, labels = [1]*len(coords))
    return(point_prompt)
