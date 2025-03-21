import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    if isinstance(mask_gt, Tensor):
        mask_gt = mask_gt.numpy()
    if isinstance(mask_pred, Tensor):
        mask_pred = mask_pred.numpy()

    if mask_gt.dtype != np.uint8:
        mask_gt = mask_gt.astype(np.uint8)
    if mask_pred.dtype != np.uint8:
        mask_pred = mask_pred.astype(np.uint8)

    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))


def show_seg_box(slice_idx, img, gt, segmentation, box_prompt):
    img_2d = img[..., slice_idx]
    gt_2d = gt[..., slice_idx]
    seg_2d = segmentation[..., slice_idx]
    box = box_prompt.value[slice_idx]

    img_2d = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min())

    # visualize results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_2d, cmap="gray")
    show_box(box, ax[0])
    ax[0].set_title("Input Image and Bounding Box")
    ax[1].imshow(gt_2d, cmap="gray")
    show_mask(seg_2d, ax[1])
    show_box(box, ax[1])
    ax[1].set_title("MedSAM Segmentation")
    plt.show()
    slice_dice = compute_dice(seg_2d, gt_2d)
    return slice_dice


def show_seg(slice_idx, img, gt, segmentation, pts_prompt=None, box_prompt=None):
    img_2d = img[..., slice_idx]
    gt_2d = gt[..., slice_idx]
    seg_2d = segmentation[..., slice_idx]

    img_2d = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min())

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_2d, cmap="gray")
    ax[1].imshow(gt_2d, cmap="gray")
    show_mask(seg_2d, ax[1])
    ax[1].set_title("MedSAM Segmentation")
    ax[0].set_title("Input Image and slice prompts")

    if pts_prompt is not None:
        coords = pts_prompt.coords
        slice_inds = coords[:, 2] == slice_idx
        slice_coords = coords[slice_inds, :-1].T
        ax[0].plot(slice_coords[1], slice_coords[0], "ro")
        ax[1].plot(slice_coords[1], slice_coords[0], "ro")

    if box_prompt is not None:
        box = box_prompt.value[slice_idx]
        show_box(box, ax[0])
        show_box(box, ax[0])
    plt.show()
    return compute_dice(seg_2d, gt_2d)


def show_seg_row_major(slice_idx, img, gt, segmentation, pts_prompt=None, box_prompt=None):
    img_2d = img[slice_idx, ...]
    gt_2d = gt[slice_idx, ...]
    seg_2d = segmentation[slice_idx, ...]

    img_2d = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min())

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_2d, cmap="gray")
    ax[1].imshow(gt_2d, cmap="gray")
    show_mask(seg_2d, ax[1])
    ax[1].set_title("Segmentation")
    ax[0].set_title("Input Image and Slice Prompts")

    if pts_prompt is not None:
        # coords, labels = pts_prompt.coords, pts_prompt.labels
        # slice_inds = coords[:,0] == slice_idx
        # slice_coords = coords[slice_inds,1:].T
        # ax[0].plot(slice_coords[1], slice_coords[0], 'ro')
        # ax[1].plot(slice_coords[1], slice_coords[0], 'ro')

        coords, labels = pts_prompt.coords, pts_prompt.labels
        slice_inds = coords[:, 0] == slice_idx
        slice_coords = coords[slice_inds, 1:].T
        slice_labels = labels[slice_inds]

        # Plot points with different colors based on their labels
        for x, y, label in zip(slice_coords[1], slice_coords[0], slice_labels):
            color = "go" if label == 1 else "ro"
            ax[0].plot(x, y, color)
            ax[1].plot(x, y, color)

    if box_prompt is not None:
        box = box_prompt.value[slice_idx]
        show_box(box, ax[0])
        show_box(box, ax[1])
    plt.show()
    return compute_dice(seg_2d, gt_2d)
