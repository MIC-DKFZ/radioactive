import random
import numpy as np
from skimage.morphology import disk, binary_dilation
from skimage.measure import find_contours
from skimage.measure import label as ski_label
import warnings

from radioa.prompts.prompt import Points, PromptStep
from radioa.prompts.prompt_3d import get_pos_clicks3D
from radioa.prompts.prompt_utils import get_fg_points_from_cc_centers, interpolate_points
from radioa.utils.analysis import compute_dice


# Code courtesy of Karol Gotkowski (modified)
def gen_contour_fp_scribble(
    slice_gt, slice_seg, contour_distance, disk_size_range, scribble_length, seed=None, verbose=True
):
    if seed:
        random.seed(seed)
    # Erode mask
    disk_size_range = random.randint(contour_distance + disk_size_range[0], contour_distance + disk_size_range[1])
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
    length = round(len(contour) * scribble_length)
    # Choose scribble position on contour - should lie on the false positives
    ## Find indices of contour corresponding to a false positive
    rounded_coords = np.round(contour).astype(int)
    values_at_contour_points = slice_seg[rounded_coords[:, 0], rounded_coords[:, 1]]
    contour_fp_inds = np.where(values_at_contour_points == 1)[0]

    if len(contour_fp_inds) == 0:
        if verbose:
            warnings.warn("All false positives not on contour: generating fg instead.")
        return None

    scribble_pos = random.choices(contour_fp_inds)[0]
    # scribble_pos = random.choices(range(len(curvature)), curvature)[0] # Draw from whole curve
    scribble_selection = (scribble_pos - int(length / 2), scribble_pos + length - int(length / 2))
    # Extract scribble
    contour = np.take(contour, range(*scribble_selection), axis=0, mode="wrap")
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


def iterate_2d(
    inferer,
    gt,
    segmentation,
    low_res_logits,
    initial_prompt,
    pass_prev_prompts,
    scribble_length=0.2,
    contour_distance=2,
    disk_size_range=(0, 0),
    dof_bound=0,
    perf_bound=0,
    init_dof=0,
    detailed=False,
    seed=None,
    verbose=True,
):
    if seed:
        np.random.seed(seed)
        random.seed(seed)
    if not inferer.image_set:
        raise RuntimeError("Must first set image!")

    # Generate initial segmentation using seed method

    # Rename for clarity
    prompt = initial_prompt
    dof = init_dof

    # Model should not be verbose during this loop. Restore state later
    verbosity = inferer.verbose
    inferer.verbose = False

    # Obtain low res masks for interactivity
    slices_inferred = np.unique(prompt.coords[:, 2])

    # Flag for calculating dof
    has_generated_positive_prompt = False

    # Tracking variables
    prompts = [prompt]
    segmentations = [segmentation.copy()]
    dice_scores = [compute_dice(segmentation, gt)]
    if verbose:
        print(dice_scores[-1])
    max_fp_idxs = []
    dofs = [init_dof]

    # Conditions to stop
    num_iter = 0
    dice_condition_met = False
    dof_condition_met = False

    while not (dice_condition_met and dof_condition_met) and num_iter <= 10:
        # Determine whether to give positive prompts or attempt negative prompt
        fn_mask = (segmentation == 0) & (gt == 1)
        fn_count = np.sum(fn_mask)

        fg_count = np.sum(gt)

        generate_positive_prompts_prob = (
            fn_count / fg_count
        )  # Generate positive prompts when much of the foreground isn't segmented
        generate_positive_prompts = np.random.binomial(1, generate_positive_prompts_prob)

        if not generate_positive_prompts:
            # Obtain contour scribble on worst sagittal slice
            fp_mask = (segmentation == 1) & (gt == 0)
            axis = 1  # Can extend to also check when fixing axis 2
            fp_sums = np.sum(fp_mask, axis=tuple({0, 1, 2} - {axis}))
            max_fp_idx = np.argmax(fp_sums)
            max_fp_idxs.append(max_fp_idx)  # For tracking
            max_fp_slice = gt[:, max_fp_idx]
            slice_seg = segmentation[:, max_fp_idx]

            if not np.any(
                max_fp_slice
            ):  # There is no gt in the slice, but lots of fps. For now just draw a vertical line down the column with the most fps
                fp_mask = slice_seg  # All segmented cells are false positives
                fp_per_column = np.sum(fp_mask, axis=0)
                max_fp_column = np.argmax(fp_per_column)
                scribble = np.zeros_like(slice_seg)
                scribble[:, max_fp_column] = 1
            else:
                scribble = gen_contour_fp_scribble(
                    max_fp_slice,
                    slice_seg,
                    contour_distance,
                    disk_size_range,
                    scribble_length,
                    seed=seed,
                    verbose=False,
                )

            if scribble is None:
                generate_positive_prompts = True  # Give random negative click instaed
            else:  # Otherwise subset scribble to false positives  to generate new prompt
                scribble_coords = np.where(scribble)
                scribble_coords = np.array(scribble_coords).T

                # Obtain false positive points and make new prompt
                is_fp_mask = slice_seg[tuple(scribble_coords.T)].astype(bool)
                fp_coords = scribble_coords[is_fp_mask]

                ## Position fp_coords back into original 3d coordinate system
                missing_axis = np.repeat(max_fp_idx, len(fp_coords))
                fp_coords_3d = np.vstack([fp_coords[:, 0], missing_axis, fp_coords[:, 1]]).T
                fp_coords_3d = fp_coords_3d[:, [2, 1, 0]]  # zyx -> xyz
                improve_slices = np.unique(fp_coords_3d[:, 2])
                dof += 3 * 4  # To dicuss: assume drawing a scribble is as difficult as drawing four points
                dofs.append(dof)

                if pass_prev_prompts:  # new prompt includes old prompts
                    ## Add to old prompt
                    coords = np.concatenate([prompt.coords, fp_coords_3d], axis=0)
                    labels = np.concatenate([prompt.labels, [0] * len(fp_coords_3d)])
                    prompt = PromptStep(point_prompts=(coords, labels))

                    ## Subset to prompts only on the slices with new prompts
                    fix_slice_mask = np.isin(prompt.coords[:, 2], improve_slices)
                    new_prompt = PromptStep(point_prompts=(coords[fix_slice_mask], labels[fix_slice_mask]))
                else:
                    new_prompt = PromptStep(point_prompts=(fp_coords_3d, [0] * len(fp_coords_3d)))

        if generate_positive_prompts:
            if not has_generated_positive_prompt:
                dof += 6  # If first time generating positive prompts, generate a bottom and top point, taking 4 degrees of freedom: (4 dof even though there are 6 coordinates since the coordinate of the lowest and highest slice is fixed)
                bottom_seed_prompt, _, top_seed_prompt = get_fg_points_from_cc_centers(gt, 3)
                has_generated_positive_prompt = True

            # Find fp coord from the middle axial range of the image
            lower, upper = np.percentile(slices_inferred, [30, 70])
            fp_coords = np.vstack(np.where(fn_mask)).T
            middle_mask = (lower < fp_coords[:, 0]) & (
                fp_coords[:, 0] < upper
            )  # Mask to determine which false negatives lie between the 30th to 70th percentile
            if np.sum(middle_mask) == 0:
                middle_mask = np.ones(
                    len(fp_coords), bool
                )  # If there are no false negatives in the middle, draw from all coordinates (unlikely given that there must be many)
            fp_coords = fp_coords[middle_mask, :]
            new_middle_seed_prompt = fp_coords[np.random.choice(len(fp_coords), 1)]
            dof += 3
            dofs.append(dof)

            # Interpolate linearly from botom_seed-prompt to top_seed_prompt through the new middle prompt to get new positive prompts
            new_seed_prompt = np.vstack([bottom_seed_prompt, new_middle_seed_prompt, top_seed_prompt])
            new_coords = interpolate_points(new_seed_prompt, kind="linear").astype(int)
            new_coords = new_coords[:, [2, 1, 0]]  # zyx -> xyz

            if pass_prev_prompts:
                # Add to old prompt
                coords = np.concatenate([prompt.coords, new_coords], axis=0)
                labels = np.concatenate([prompt.labels, [1] * len(new_coords)])
                new_prompt = PromptStep(point_prompts=(coords, labels))
                new_prompt = prompt
            else:
                prompt = PromptStep(point_prompts=(new_coords, [1] * len(new_coords)))
                new_prompt = prompt

            improve_slices = slices_inferred  # improve all slices

        # Generate new segmentation and integrate into old one
        new_seg, low_res_logits, _ = inferer.predict(new_prompt, low_res_logits, transform=False)
        prompts.append(new_prompt)
        segmentation[improve_slices] = new_seg[improve_slices]
        segmentations.append(segmentation.copy())

        # Update the trackers
        low_res_logits.update(low_res_logits)
        dice_scores.append(compute_dice(segmentation, gt))
        if verbose:
            print(dice_scores[-1])

        # Check break conditions
        if dof >= dof_bound:
            dof_condition_met = True

        if dice_scores[-1] >= perf_bound:
            dice_condition_met = True

        num_iter += 1

    # Reset verbosity
    inferer.verbose = verbosity

    # Return variables with chosen degree of detail
    if detailed:
        return dice_scores, dofs, segmentations, prompts
    else:
        return dice_scores, dofs


def iterate_3d(
    inferer, gt_unprocessed, pass_prev_prompts, perf_bound, dof_bound, seed=None, detailed=False, sammed3d=False
):
    if seed is not None:
        np.random.seed(seed)

    if not inferer.image_set:
        raise RuntimeError("Must first set image!")

    n = 1

    dice_scores = []
    segmentations = []
    prompts = []
    dofs = []
    num_iter = 0
    dof = 0

    # Obtain initial segmentation

    prompt = get_pos_clicks3D(gt_unprocessed, n, seed=seed)
    segmentation, logits, _ = inferer.predict(prompt, store_patching=True, transform=False)

    gt_to_compare = gt_unprocessed
    # For SAMMed3D, must sample for new prompts only in a 128x128x128 crop around the initial prompt.
    if sammed3d:
        gt_to_compare = np.zeros_like(gt_unprocessed)
        seed_coords = prompt.coords[0]
        low = np.maximum(seed_coords - 64, 0)
        high = np.minimum(seed_coords + 64, gt_unprocessed.shape)
        gt_to_compare[low[0] : high[0], low[1] : high[1], low[2] : high[2]] = gt_unprocessed[
            low[0] : high[0], low[1] : high[1], low[2] : high[2]
        ]

    prompts.append(prompt)
    dof += 3
    segmentations.append(segmentation.copy())
    dice_scores.append(compute_dice(segmentation, gt_unprocessed))

    dice_condition_met, dof_condition_met = False, False
    while not (dice_condition_met and dof_condition_met) and num_iter <= 10:

        # Obtain new prompt
        misclassifieds = np.vstack(np.where(segmentation != gt_to_compare)).T
        sampled_ind = np.random.randint(len(misclassifieds))
        sampled_coords = [misclassifieds[sampled_ind]]
        sampled_labels = [gt_to_compare[tuple(sampled_coords[0])]]

        if pass_prev_prompts:
            sampled_coords = np.vstack((prompt.coords, sampled_coords))
            sampled_labels = np.hstack((prompt.labels, sampled_labels))

        prompt = Points(coords=sampled_coords, labels=sampled_labels)
        prompts.append(prompt)

        segmentation, logits, _ = inferer.predict(
            prompt, use_stored_patching=True, prev_low_res_logits=logits, transform=False
        )

        prompts.append(prompt)
        dof += 3
        segmentations.append(segmentation.copy())
        dice_scores.append(compute_dice(segmentation, gt_unprocessed))

        # Check break conditions
        if dof >= dof_bound:
            dof_condition_met = True

        if dice_scores[-1] >= perf_bound:
            dice_condition_met = True

        num_iter += 1

    if detailed:
        return dice_scores, dofs, segmentations, prompts
    else:
        return dice_scores, dofs
