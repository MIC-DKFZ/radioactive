from loguru import logger
import numpy as np
from copy import deepcopy

# from skimage.morphology import dilation, ball
from skimage.measure import label
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

from intrab.model.inferer import Inferer
from intrab.prompts.prompt import Points, PromptStep

# def get_neg_clicks_3D(gt, n_clicks, border_distance = 10): # Warning: dilation function is VERY slow! ~ 13 seconds on my machine
#     struct_element = ball(border_distance)
#     volume_dilated = dilation(gt, struct_element)

#     border_region = volume_dilated - gt

#     volume_border = np.where(border_region)
#     volume_border = np.array(volume_border).T
#     n_border_voxels = volume_border.shape[0]

#     if n_border_voxels < n_clicks:
#         raise RuntimeError(f'More background points were requested than the number of border voxels in the volume ({n_clicks} vs {n_border_voxels})')

#     point_indices = np.random.choice(n_border_voxels, size = n_clicks, replace = False)
#     neg_coords = volume_border[point_indices]  # change from triple of arrays format to list of triples format # change from triple of arrays format to list of triples format

#     neg_coords = Points(coords = neg_coords, labels =  [0]*len(neg_coords))
#     return(neg_coords)


def increment(values: np.ndarray, upwards: bool):
    return (values + 1) if upwards else (values - 1)


def get_actual_indices_of_selected_indices(
    proposed_slices: list[int], z_indices_with_foreground: list[int]
) -> list[int]:
    """Makes sure that the proposed slices are actually containing foreground.
    This is only necessary for cases where organs are not one continuous volume, but have gaps in between slices."""
    actually_selected_slices = []
    for sel_sli in proposed_slices:
        if sel_sli in z_indices_with_foreground:
            actually_selected_slices.append(sel_sli)
        else:
            actually_selected_slices.append(
                z_indices_with_foreground[np.argmin(np.abs(z_indices_with_foreground - sel_sli))]
            )
    return actually_selected_slices


def get_slices_to_do(
    current_slice, slices_to_infer, upwards: bool, include_start: bool = False, include_end: bool = False
):
    start_offset = 0 if include_start else 1
    end_offset = 1 if include_end else 0

    if upwards:
        max_slice = slices_to_infer.max()
        return list(range(current_slice + start_offset, max_slice + end_offset))
    else:
        min_slice = slices_to_infer.min()
        return list(range(current_slice - start_offset, min_slice - end_offset, -1))


def get_2d_bbox_of_gt_slice(mask):  # Bbox function only used in this function
    i_any = np.any(mask, axis=1)
    j_any = np.any(mask, axis=0)
    i_min, i_max = np.where(i_any)[0][[0, -1]]
    j_min, j_max = np.where(j_any)[0][[0, -1]]
    bb_min = np.array([i_min, j_min])
    bb_max = np.array([i_max, j_max])

    return bb_min, bb_max


def get_pos_clicks2D_row_major(gt, n_clicks, seed=None):
    """
    Receives a groundtruth and a number of clicks (per slice) and generates a dictionary of point coordinates for each slice
    """

    if seed is not None:
        np.random.seed(seed)
    volume_fg = np.where(gt == 1)  # Get foreground indices (formatted as triple of arrays)
    volume_fg = np.array(volume_fg).T  # Reformat to numpy array of shape n_fg_voxels x 3

    fg_slices = np.unique(
        volume_fg[:, 0]
    )  # Obtain superior axis slices which have foreground before reformating indices

    pos_coords = np.empty(shape=(0, 3), dtype=int)
    warning_zs = []  # track slices without enough foreground/border, if any should exist

    for slice_index in fg_slices:
        ## Foreground points
        slice = gt[slice_index, :, :]
        slice_fg = np.where(slice)
        slice_fg = np.array(slice_fg).T

        n_fg_pixels = len(slice_fg)
        if n_fg_pixels >= n_clicks:
            point_indices = np.random.choice(n_fg_pixels, size=n_clicks, replace=False)
        else:
            # In this case, take all foreground pixels and then obtain some duplicate points by sampling with replacement additionally
            warning_zs.append(f"z = {slice_index}, n foreground = n_fg_pixels")
            point_indices = np.concatenate(
                [np.arange(n_fg_pixels), np.random.choice(n_fg_pixels, size=n_clicks - n_fg_pixels, replace=True)]
            )

        pos_clicks_slice = slice_fg[point_indices]
        z_col = np.full((n_clicks, 1), slice_index)  # create z column to add
        pos_clicks_slice = np.hstack([z_col, pos_clicks_slice])
        pos_coords = np.vstack([pos_coords, pos_clicks_slice])

    pos_coords = pos_coords[:, [2, 1, 0]]  # gt is in row-major zyx, so need to reorder to get points in xyz.
    point_prompt = PromptStep(point_prompts=(pos_coords, np.array([1] * len(pos_coords))))
    return point_prompt


def _get_bbox3d(mask_volume: np.ndarray):
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
    bbox3d = _get_bbox3d(mask_volume)

    slices_to_infer = np.arange(bbox3d[0][0], bbox3d[1][0])  # gt is in ZYX format, so index 0 are the axial slices
    box_dict = {
        slice_idx: np.array((bbox3d[0][2], bbox3d[0][1], bbox3d[1][2], bbox3d[1][1])) for slice_idx in slices_to_infer
    }  # reverse order to get xyxy
    box_prompt = PromptStep(box_prompts=box_dict)
    return box_prompt


def _get_bbox2d(mask):
    i_any = np.any(mask, axis=1)
    j_any = np.any(mask, axis=0)
    i_min, i_max = np.where(i_any)[0][[0, -1]]
    j_min, j_max = np.where(j_any)[0][[0, -1]]
    bb_min = np.array([i_min, j_min])
    bb_max = np.array([i_max, j_max]) + 1

    coord_list = np.concatenate([bb_min, bb_max])  # SAM wants row-major x0y0x1y1
    return coord_list


def _get_bbox2d_row_major(mask):
    """
    Mask must be in row major roder
    """
    i_any = np.any(mask, axis=1)
    j_any = np.any(mask, axis=0)
    i_min, i_max = np.where(i_any)[0][[0, -1]]
    j_min, j_max = np.where(j_any)[0][[0, -1]]
    bb_min = np.array([j_min, i_min])
    bb_max = np.array([j_max, i_max]) + 1

    coord_list = np.concatenate([bb_min, bb_max])
    return coord_list


def get_minimal_boxes_row_major(gt, delta_x=0, delta_y=0):
    """
    gt must be in row-major ZYX order.
    Get bounding boxes of the foreground per slice. delta_x, delta_y enlargen the box
    in the respective dimensions
    """
    slices_to_infer = np.where(np.any(gt, axis=(1, 2)))[0]  # index 0 since a tuple of length 1 is returned
    box_dict = {slice_idx: _get_bbox2d_row_major(gt[slice_idx, :, :]) for slice_idx in slices_to_infer}
    box_dict = {
        slice_idx: box + np.array([-delta_x, -delta_y, delta_x, delta_y]) for slice_idx, box in box_dict.items()
    }
    box_prompt = PromptStep(box_prompts=box_dict)
    return box_prompt


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
    assert labels.max() != 0  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC.astype(int)


def get_fg_point_from_cc_center(gt_slice: np.ndarray) -> np.ndarray:
    """
    Extract the nearest foreground point from the center of mass of the largest connected component.

    :param gt_slice: A 2D binary mask.
    :return: The nearest foreground point index [x, y].  <-- To be verified
    """
    largest_cc = get_largest_CC(gt_slice)
    fg_indices = np.where(largest_cc)
    fg_mean_indices = np.mean(fg_indices, axis=1).round().astype(int)
    # ToDo: verify x,y are not flipped.
    nearest_fg_point = get_nearest_fg_point(fg_mean_indices, largest_cc)
    return nearest_fg_point


def get_fg_points_from_cc_centers(gt, n):
    """
    Takes n equally spaced slices starting at z_min and ending at z_max, where z_min is the lowest transverse slice of gt containing fg, and z_max similarly with highest,
    finds the largest connected component of fg, takes the center of its bounding box and takes the nearest fg point. Simulates a clinician clicking in the 'center of the main mass of the roi' per slice
    """
    z_indices = np.where(np.any(gt, axis=(1, 2)))[0]
    min_z, max_z = np.min(z_indices), np.max(z_indices)
    selected_slices = np.linspace(min_z, max_z, num=n, dtype=int)
    selected_slices = get_actual_indices_of_selected_indices(selected_slices, z_indices)

    corrected_points = np.empty([n, 3], dtype=int)
    for i, z in enumerate(selected_slices):  # selected_slices:
        largest_cc = get_largest_CC(gt[z])
        bbox_min, bbox_max = get_2d_bbox_of_gt_slice(largest_cc)

        slice_fg_center = np.vstack([bbox_min, bbox_max]).mean(axis=0).round()

        nearest_fg_point = get_nearest_fg_point(slice_fg_center, largest_cc)
        corrected_points[i] = np.concatenate([[z], nearest_fg_point])

    return corrected_points


def interpolate_points(points, kind="linear"):
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


def get_middle_seed_point(fn_mask, slices_inferred):
    lower, upper = np.percentile(slices_inferred, [30, 70])
    fp_coords = np.vstack(np.where(fn_mask)).T
    middle_mask = (lower < fp_coords[:, 0]) & (
        fp_coords[:, 0] < upper
    )  # Mask to determine which false negatives lie between the 30th to 70th percentile
    if np.sum(middle_mask) == 0:
        logger.info("Failed to generate prompt in middle 40 percent of the volume. This may be worth checking out.")
        middle_mask = np.ones(
            len(fp_coords), bool
        )  # If there are no false negatives in the middle, draw from all coordinates (unlikely given that there must be many)

    fp_coords = fp_coords[middle_mask, :]
    new_middle_seed_prompt = fp_coords[np.random.choice(len(fp_coords), 1)]

    return new_middle_seed_prompt


def point_interpolation(gt, n_slices, interpolation="linear"):
    """
    Simulates a clinician clicking in the 'center of the main mass of the roi' per slice.
    Between these slices, the points are interpolated using linear or cubic spline interpolation.

    :param gt: The ground truth volume.
    :param n_slices: The number of slices to interpolate points between.
    """
    simulated_clicks = get_fg_points_from_cc_centers(gt, n_slices)
    coords = interpolate_points(simulated_clicks, kind=interpolation).astype(int)
    coords = coords[:, [2, 1, 0]]  # Gt is in row-major; need to reorder to xyz
    point_prompt = PromptStep(point_prompts=(coords, np.array([1] * len(coords))))
    return point_prompt


# ToDo: Add an additional function that takes the center-of-mass of the largest component.
#   Or a random point from a strongly eroded version of the groundtruth.


def get_fg_points_from_slice(slice: np.ndarray, n_clicks: int, slice_index: int, seed=None):
    """
    Gets random forground points from a slice.
    If n_clicks > n_fg_pixels, will return all n_fg_pixels -- May be lower than n_clicks.

    :param slice: A 2D binary mask.
    :param n_clicks: (int) The number of clicks to generate.
    :return: (np.ndarray n x 3) The coordinates of the foreground points in (x y z) format.
    """
    if seed:
        np.random.seed(seed)
    slice_fg = np.where(slice)
    slice_fg = np.array(slice_fg).T

    n_fg_pixels = len(slice_fg)
    actual_clicks = min(n_clicks, n_fg_pixels)
    point_indices = np.random.choice(n_fg_pixels, size=actual_clicks, replace=False)

    pos_clicks_slice = slice_fg[point_indices]

    z_col = np.full((actual_clicks, 1), slice_index)
    pos_coords = np.hstack([z_col, pos_clicks_slice])
    pos_coords = pos_coords[:, [2, 1, 0]]  # zyx -> xyz

    return pos_coords


def get_seed_boxes(gt, n) -> PromptStep:
    z_indices = np.where(np.any(gt, axis=(1, 2)))[0]
    min_z, max_z = np.min(z_indices), np.max(z_indices)
    if n > 1:
        selected_slices = np.linspace(min_z, max_z, num=n, dtype=int)
        # This can contain duplicates or slices without foreground (if not continuous foreground)
        actually_selected_slices = get_actual_indices_of_selected_indices(selected_slices, z_indices)

    if n == 1:
        median_index = int(np.median(z_indices))
        if median_index not in z_indices:
            median_index = z_indices[np.argmin(np.abs(median_index - z_indices))]
        actually_selected_slices = (median_index,)
    # ToDo: Make sure the boxes are actually interpolated from within the z_indices, where there is foreground to pull from

    # This de-duplicates the prompts of the same key.
    bbox_dict = {slice_idx: _get_bbox2d_row_major(gt[slice_idx]) for slice_idx in actually_selected_slices}

    return PromptStep(box_prompts=bbox_dict)


def box_interpolation(seed_boxes: PromptStep):
    """
    Takes n equally spaced slices starting at z_min and ending at z_max, where z_min is the lowest transverse slice of gt containing fg, and z_max similarly with highest,
    finds the largest connected component of fg, takes the center of its bounding box and takes the nearest fg point. Simulates a clinician clicking in the 'center of the main mass of the roi' per slice
    """

    bbox_mins = np.array([(slice_idx, bbox[0], bbox[1]) for slice_idx, bbox in seed_boxes.boxes.items()])
    bbox_mins_interpolated = interpolate_points(bbox_mins)

    bbox_maxs = np.array([(slice_idx, bbox[2], bbox[3]) for slice_idx, bbox in seed_boxes.boxes.items()])
    bbox_maxs_interpolated = interpolate_points(bbox_maxs)

    bbox_np_interpolated = np.concatenate((bbox_mins_interpolated, bbox_maxs_interpolated[:, 1:]), axis=1).astype(int)

    bbox_dict_interpolated = {row[0]: row[1:] for row in bbox_np_interpolated}
    box_prompt = PromptStep(box_prompts=bbox_dict_interpolated)

    return box_prompt


def get_seed_point(gt, n_clicks, seed) -> PromptStep:
    slices_to_infer = np.where(np.any(gt, axis=(1, 2)))[0]
    # ToDo: Maybe pull not always the median slice.
    middle_idx = np.median(slices_to_infer).astype(int)

    pos_coords = get_fg_points_from_slice(gt[middle_idx], n_clicks, middle_idx, seed)

    ## Put coords in 3d context

    pts_prompt = PromptStep(point_prompts=(pos_coords, [1] * pos_coords.shape[0]))

    return pts_prompt


def propagate_point(
    inferer: Inferer, seed_prompt: PromptStep, slices_to_infer: list[int], upwards: bool, seed: int, n_clicks=5
) -> tuple[np.ndarray, np.ndarray]:

    # assert len(seed_prompt.coords) == 1, "Seed point must contain only one point prompt."
    start_slice = seed_prompt.coords[0][-1]
    # The todo slices INCLUDES the seed slice. This is different from the box propagation.
    slices_todo = get_slices_to_do(start_slice, slices_to_infer, upwards, include_start=True, include_end=False)

    # Get the coordinates of the next point prompt -- Used to get the next slice's Prompt
    all_coords = [seed_prompt.coords]
    all_labels = [seed_prompt.labels]
    current_prompt = seed_prompt
    for slice_id in slices_todo:
        current_seg_nib, _, _ = inferer.predict(current_prompt)
        current_seg = inferer.transform_to_model_coords(current_seg_nib, is_seg=True)[0]
        coords_xyz = get_fg_points_from_slice(
            current_seg[slice_id], n_clicks=n_clicks, slice_index=slice_id, seed=seed
        )
        coords_xyz[:, -1] = increment(coords_xyz[:, -1], upwards)  # Increment the z-coordinate
        labels = np.ones_like(coords_xyz[:, 0])
        all_coords.append(coords_xyz)
        all_labels.append(labels)
        current_prompt = PromptStep(Points(coords=coords_xyz, labels=labels))
    return np.concatenate(all_coords, axis=0), np.concatenate(all_labels, axis=0)


def point_propagation(
    inferer: Inferer, seed_prompt: PromptStep, slices_to_infer, seed=None, n_clicks: int = 5
) -> PromptStep:

    upwards_coords, upward_labels = propagate_point(
        inferer, seed_prompt, slices_to_infer, upwards=True, seed=seed, n_clicks=n_clicks
    )
    downward_coords, downward_labels = propagate_point(
        inferer, seed_prompt, slices_to_infer, upwards=False, seed=seed, n_clicks=n_clicks
    )
    # Skip the first slice as it it the seed slice for both upwards and downwards
    all_coords = np.concatenate((upwards_coords, downward_coords[n_clicks:]), axis=0)
    all_labels = np.concatenate((upward_labels, downward_labels[n_clicks:]), axis=0)
    return PromptStep(Points(coords=all_coords, labels=all_labels))


def get_seed_box(gt):
    slices_to_infer = np.where(np.any(gt, axis=(1, 2)))[0]
    # ToDo: Maybe pull not always the median slice.
    #   Make this jittered between 30-70 percent of the volume
    middle_idx = np.median(slices_to_infer).astype(int)

    bbox_slice = _get_bbox2d_row_major(gt[middle_idx])
    box_dict = {middle_idx: bbox_slice}
    box_prompt = PromptStep(box_prompts=box_dict)

    return box_prompt


def propagate_box(inferer: Inferer, seed_box: PromptStep, slices_to_infer: list[int], upwards: bool) -> dict:
    """"""
    # if upwards:

    #     def get_slices_to_do(current_slice, slices_to_infer):
    #         max_slice = slices_to_infer.max()
    #         return list(range(current_slice + 1, max_slice + 1))

    # else:

    #     def get_slices_to_do(current_slice, slices_to_infer):
    #         min_slice = slices_to_infer.min()
    #         return list(range(current_slice - 1, min_slice - 1, -1))

    assert len(seed_box.boxes) == 1, "Seed box must contain only one box prompt."
    start_slice = list(seed_box.boxes.keys())[0]
    slices_todo = get_slices_to_do(start_slice, slices_to_infer, upwards, include_start=False, include_end=True)
    if len(slices_todo) == 0:
        return {}
    all_boxes = {}
    # This is the prompt given for the median slice. It is directly transferred to the one above or below.
    #   The segmentation is used to get the box for the slice after, if it should be inferred.
    current_prompt = seed_box
    for cnt, slice_idx in enumerate(slices_todo):
        # We extract the box coordinates either from the segmentation or the initial box.
        current_box = list(current_prompt.boxes.values())[0]
        current_prompt = PromptStep(box_prompts={slice_idx: current_box})
        # Save the current box prompt
        all_boxes.update(current_prompt.boxes)

        # If there is no next slice to infer, we can stop here.
        if cnt != len(slices_todo) - 1:
            segmentation = inferer.predict(current_prompt)[-1]
            # segmentation, _ = inferer.transform_to_model_coords(segmentation, is_seg=True)
            if np.all(segmentation[slice_idx] == 0):  # Terminate if no fg generated
                logger.debug("No prediction despite prompt given. Stopping propagation.")
                break
            bbox_slice = _get_bbox2d_row_major(segmentation)
            current_prompt = PromptStep(box_prompts={0: bbox_slice})
            # The slice index is 0 deliberately as it is never needed.
    return all_boxes


# ToDo: Add a test that verifies that the segmentation of the
#   box prompts are the same when run twice.
def box_propagation(inferer: Inferer, seed_box: PromptStep, slices_to_infer) -> PromptStep:
    """
    Propagate a seed box prompt through the slices to infer in a 3D volume.
    Returns all the box prompts that were generated to pass to a single predict call.
    """

    # Initialise segmentation to store total result
    # segmentation = np.zeros_like(inferer.img).astype(np.uint8)

    # box_prompt = deepcopy(seed_box)
    # all_boxes =   # Update throughout the loops to keep track of all box prompts
    # middle_idx = np.median(slices_to_infer).astype(int)
    upwards_boxes = propagate_box(inferer, seed_box, slices_to_infer, upwards=True)
    downward_boxes = propagate_box(inferer, seed_box, slices_to_infer, upwards=False)
    # Combine all boxes
    all_boxes = {**seed_box.boxes, **upwards_boxes, **downward_boxes}
    return PromptStep(box_prompts=all_boxes)
