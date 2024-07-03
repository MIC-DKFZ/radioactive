import random
import numpy as np
from skimage.morphology import disk, binary_dilation
from skimage.measure import find_contours
from skimage.measure import label as ski_label
import warnings
from skimage.filters import gaussian

# Code courtesy of Karol Gotkowski
def normalize(x, source_limits=None, target_limits=None):
    if source_limits is None:
        source_limits = (x.min(), x.max())

    if target_limits is None:
        target_limits = (0, 1)

    if source_limits[0] == source_limits[1] or target_limits[0] == target_limits[1]:
        return x * 0
    else:
        x_std = (x - source_limits[0]) / (source_limits[1] - source_limits[0])
        x_scaled = x_std * (target_limits[1] - target_limits[0]) + target_limits[0]
        return x_scaled
    
# Code courtesy of Karol Gotkowski
def compute_curvature(contour):
    dx = np.gradient(contour[:, 0])
    dy = np.gradient(contour[:, 1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = (dx * d2y - dy * d2x) / np.power(dx**2 + dy**2, 1.5)
    curvature = np.abs(curvature)
    curvature = gaussian(curvature)
    curvature = normalize(curvature)
    return curvature

# Code courtesy of Karol Gotkowski
def gen_contour_scribble(mask, contour_distance, disk_size_range, scribble_length_range, seed = None): 
    if seed:
        random.seed(seed)
    # Erode mask
    disk_size_range = random.randint(contour_distance+disk_size_range[0], contour_distance+disk_size_range[1])
    eroded_mask = binary_dilation(mask, disk(disk_size_range))
    if not np.any(np.nonzero(eroded_mask)):
        return None
    # Compute curvature of the contour
    contour = find_contours(eroded_mask)
    if len(contour) == 0:
        return None
    contour = np.concatenate(contour, axis=0)
    # Compute curvature of the contour
    curvature = compute_curvature(contour)
    # Compute scribble length
    min_length = int(len(contour)*scribble_length_range[0])
    max_length = int(len(contour)*scribble_length_range[1])
    min_length = min_length if min_length > 0 else 1
    max_length = max_length if max_length > 0 else 1
    length = random.randint(min_length, max_length)
    # Choose scribble position on contour
    scribble_pos = random.choices(range(len(curvature)), curvature)[0] # Weighted by curvature - not necessary in my use cas
    scribble_selection = (scribble_pos-int(length/2), scribble_pos+length-int(length/2))
    # Extract scribble
    contour = np.take(contour, range(*scribble_selection), axis=0, mode='wrap')
    contour = np.round(contour).astype(np.int32)
    # Render scribble
    scribble = np.zeros_like(mask)
    scribble[contour[:, 0], contour[:, 1]] = 1
    # It is not guaranteed that the scribble is not a set of scribbles, so we remove all but the largest one
    scribble_components = ski_label(scribble)
    labels, counts = np.unique(scribble_components, return_counts=True)
    counts = counts[labels > 0]
    labels = labels[labels > 0]
    label = labels[np.argmax(counts)]
    scribble = scribble_components == label
    return scribble

def gen_contour_fp_scribble(slice_gt, slice_seg, contour_distance, disk_size_range, scribble_length, seed = None): 
    if seed:
        random.seed(seed)
    # Erode mask
    disk_size_range = random.randint(contour_distance+disk_size_range[0], contour_distance+disk_size_range[1])
    eroded_mask = binary_dilation(slice_gt, disk(disk_size_range))
    if not np.any(np.nonzero(eroded_mask)):
        return None
    # Compute curvature of the contour
    contour = find_contours(eroded_mask)
    if len(contour) == 0:
        return None
    contour = np.concatenate(contour, axis=0)
    # Compute scribble length
    # min_length = int(len(contour)*scribble_length_range[0])
    # max_length = int(len(contour)*scribble_length_range[1])
    # min_length = min_length if min_length > 0 else 1
    # max_length = max_length if max_length > 0 else 1
    # length = random.randint(min_length, max_length)
    length = round(len(contour)*scribble_length)
    # Choose scribble position on contour - should lie on the false positives
    ## Find indices of contour corresponding to a false positive
    rounded_coords = np.round(contour).astype(int)
    values_at_contour_points = slice_seg[rounded_coords[:, 0], rounded_coords[:, 1]]
    contour_fp_inds = np.where(values_at_contour_points == 1)[0]

    if len(contour_fp_inds) == 0:
        warnings.warn(' All false positives not on contour: generating fg instead.')
        return None

    scribble_pos = random.choices(contour_fp_inds)[0]
    # scribble_pos = random.choices(range(len(curvature)), curvature)[0] # Draw from whole curve
    scribble_selection = (scribble_pos-int(length/2), scribble_pos+length-int(length/2))
    # Extract scribble
    contour = np.take(contour, range(*scribble_selection), axis=0, mode='wrap')
    contour = np.round(contour).astype(np.int32)
    # Render scribble
    scribble = np.zeros_like(slice_gt)
    scribble[contour[:, 0], contour[:, 1]] = 1
    # It is not guaranteed that the scribble is not a set of scribbles, so we remove all but the largest one
    scribble_components = ski_label(scribble)
    labels, counts = np.unique(scribble_components, return_counts=True)
    counts = counts[labels > 0]
    labels = labels[labels > 0]
    label = labels[np.argmax(counts)]
    scribble = scribble_components == label
    return scribble