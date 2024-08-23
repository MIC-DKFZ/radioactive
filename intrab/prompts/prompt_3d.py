import numpy as np

from skimage.morphology import dilation, ball

from intrab.prompts.prompt import Boxes3D, Points


def get_crop_pad_center_from_points(points):
    bbox_min = points.coords.T.min(
        axis=1
    )  # Get an array of two points: the minimal and maximal vertices of the minimal cube parallel to the axes bounding the points
    bbox_max = points.coords.T.max(axis=1) + 1  # Add 1 since we'll be using this for indexing
    point_center = np.mean((bbox_min, bbox_max), axis=0)

    return point_center


def crop_pad_coords(coords, cropping_params, padding_params):
    axis_add, axis_sub = padding_params[::2], cropping_params[::2]
    coords = (
        coords + axis_add - axis_sub
    )  # same as value[:,i] = value[:,i] + axis_add[i] - axis_sub[i] iterating over i
    return coords


def get_pos_clicks3D(gt, n_clicks, seed=None):
    if seed is not None:
        np.random.seed(seed)

    volume_fg = np.where(gt == 1)  # Get foreground indices (formatted as triple of arrays)
    volume_fg = np.array(volume_fg).T  # Reformat to numpy array of shape n_fg_voxels x 3

    n_fg_voxels = len(volume_fg)

    # Error testing
    if n_fg_voxels == 0:
        raise RuntimeError(
            f"No foreground voxels found! Check that the supplied label is a binary segmentation mask with foreground coded as 1"
        )

    if n_fg_voxels < n_clicks:
        raise RuntimeError(
            f"More foreground points were requested than the number of foreground voxels in the volume"
        )

    point_indices = np.random.choice(n_fg_voxels, size=n_clicks, replace=False)
    pos_coords = volume_fg[point_indices]
    pos_coords = pos_coords[:, ::-1]  # Assume gt is in row major zyx, need to reverse order
    pos_coords = Points(coords=pos_coords, labels=[1] * len(pos_coords))
    return pos_coords


def get_neg_clicks_3D(
    gt, n_clicks, border_distance=10
):  # Warning: dilation function is VERY slow! ~ 13 seconds on my machine
    struct_element = ball(border_distance)
    volume_dilated = dilation(gt, struct_element)

    border_region = volume_dilated - gt

    volume_border = np.where(border_region)
    volume_border = np.array(volume_border).T
    n_border_voxels = volume_border.shape[0]

    if n_border_voxels < n_clicks:
        raise RuntimeError(
            f"More background points were requested than the number of border voxels in the volume ({n_clicks} vs {n_border_voxels})"
        )

    point_indices = np.random.choice(n_border_voxels, size=n_clicks, replace=False)
    neg_coords = volume_border[
        point_indices
    ]  # change from triple of arrays format to list of triples format # change from triple of arrays format to list of triples format

    neg_coords = Points(coords=neg_coords, labels=[0] * len(neg_coords))
    return neg_coords


def get_bbox3d(mask_volume: np.ndarray, delta=0):
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
    bb_min = np.array([i_min, j_min, k_min]) - delta
    bb_max = np.array([i_max, j_max, k_max]) + delta

    # Assume gt is in row major zyx - reverse order of coordinates
    bb_min = bb_min[::-1]
    bb_max = bb_max[::-1]
    bb = Boxes3D(bb_min, bb_max)
    return bb
