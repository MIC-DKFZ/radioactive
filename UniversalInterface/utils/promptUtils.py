import matplotlib.pyplot as plt
import torch
import numpy as np
from skimage.morphology import dilation, ball, disk
from .base_classes import Points, Boxes2d
from skimage.measure import label
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

def get_crop_pad_center_from_points(points):
    bbox_min = points.value['coords'].T.min(axis = 1) # Get an array of two points: the minimal and maximal vertices of the minimal cube parallel to the axes bounding the points
    bbox_max = points.value['coords'].T.max(axis = 1) + 1 # Add 1 since we'll be using this for indexing 
    point_center = np.mean((bbox_min, bbox_max), axis = 0)  

    return(point_center)

def crop_pad_coords(coords, cropping_params, padding_params):
    axis_add, axis_sub = padding_params[::2], cropping_params[::2] 
    coords = coords + axis_add - axis_sub # same as value[:,i] = value[:,i] + axis_add[i] - axis_sub[i] iterating over i
    return(coords)

def get_pos_clicks3D(gt, n_clicks, seed = None):
    if seed is not None:
        np.random.seed(seed)
        
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
    j_any = np.any(mask, axis=0)
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

def get_minimal_boxes_row_major(gt, delta_x, delta_y):
    '''
    gt must be in row-major ZYX order.
    Get bounding boxes of the foreground per slice. delta_x, delta_y enlargen the box
    in the respective dimensions
    '''
    slices_to_infer = np.where(np.any(gt, axis=(1,2))) # index 0 since a tuple of length 1 is returned
    slices_to_infer = slices_to_infer[0]
    box_dict = {slice_idx: get_bbox2d_row_major(gt[slice_idx,:,:]) for slice_idx in slices_to_infer} # Transpose to get coords in row-major format
    box_dict = {slice_idx: box + np.array([-delta_x, -delta_y, delta_x, delta_y]) for slice_idx, box in box_dict.items()}
    return(Boxes2d(box_dict))

# Have to rework for new row-major order
# def get_3d_box_for_2d(gt, delta_x, delta_y):
#     '''
#     Finds 3d bounding box over the volume and returns it in a 2d prompt
#     '''
#     box_3d = get_bbox3d(gt.transpose(2,1,0)) # Permute to get coords in row major format
#     box_2d = np.concatenate([box_3d[0][-2:], box_3d[1][-2:]])
#     box_2d = box_2d + np.array([-delta_x, -delta_y, delta_x, delta_y])
#     box_dict = {slice_idx: box_2d for slice_idx in range(box_3d[0][2], box_3d[1][2])}
#     return(Boxes2d(box_dict))

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
