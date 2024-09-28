import numpy as np


class Points:
    def __init__(self, coords, labels):

        self.coords = np.atleast_2d(np.array(coords))  # Nx3  [x, y, z]
        self.labels = np.array(labels)  # N    [0 or 1]

    def get_slices_to_infer(self):
        unique_zs = set(self.coords[:, 2])
        return unique_zs


class Boxes:
    def __init__(self, value):
        """
        self.value must be a dictionary with items {slice_number:bounding box for slice}
        """
        self.value = value

    def get_slices_to_infer(self):
        return list(self.value.keys())

    def keys(self):
        return self.value.keys()


class Boxes3D:
    def __init__(self, bbox_min, bbox_max):
        assert len(bbox_min) == 3 and len(bbox_max) == 3
        assert np.all(bbox_min <= bbox_max)
        self.min, self.max = np.array(bbox_min), np.array(bbox_max)
        self.bbox = (self.min, self.max)

    def __getitem__(self, idx):
        return self.bbox[idx]


class PromptStep:
    def __init__(
        self,
        point_prompts: Points | tuple[np.ndarray, np.ndarray] | None = None,
        box_prompts: Boxes | dict[int, np.ndarray] | None = None,
        mask_prompts: dict[int, np.ndarray] | np.ndarray | None = None 
    ):
        """
        Initialise a promptstep with (some of) points, boxes and masks
        Points must be supplied as a pair (coords, labels), with coords nx3 containing the coordinates (xyz) of the prompts, or as a Points object
        Boxes must be supplied as a dictionary of z_slice: (x_min, y_min, x_max, y_max) or as a Boxes object
        """
        # Default initiation
        self.coords: np.ndarray | None = None
        self.labels: np.ndarray | None = None

        self.boxes: dict[int, np.ndarray] | None = None # ToDo change to None, propagate necessary changes

        self.masks: np.ndarray | None = None

        self.has_points = self.has_boxes = self.has_masks = False

        self.set_points(point_prompts)
        self.set_boxes(box_prompts)
        self.set_masks(mask_prompts)


    def set_points(self, point_prompts: Points | tuple[np.ndarray, np.ndarray] | None):
        assert isinstance(point_prompts, (Points, tuple, type(None))), f'must be supplied as a pair (coords, labels), with coords nx3 containing the coordinates (xyz) of the prompts, or as a Points object. Got {type(point_prompts)}'

        if isinstance(point_prompts, Points):
            assert len(point_prompts.coords) == len(point_prompts.labels), 'Labels and coordinates do not match up - Not the same number of each.'
            self.coords, self.labels = point_prompts.coords, point_prompts.labels
            self.has_points = True
        elif isinstance(point_prompts, tuple):
            assert len(point_prompts[0]) == len(point_prompts[1]), 'Labels and coordinates do not match up - Not the same number of each.'
            self.coords, self.labels = np.atleast_2d(np.array(point_prompts[0])), np.array(point_prompts[1])
            self.has_points = True
        elif point_prompts is None:
            self.coords, self.labels = None, None
            self.has_points = False

    def set_boxes(self, box_prompts: Boxes | dict[int, np.ndarray] | None):
        assert isinstance(box_prompts, (Boxes, dict, type(None))), f'Boxes must be supplied as a dictionary of z_slice: (x_min, y_min, x_max, y_max) or as a Boxes object. Got {type(box_prompts)}'
        if isinstance(box_prompts, Boxes):
            self.boxes = self.boxes.value
            self.has_boxes = True
        elif isinstance(box_prompts, dict):
            self.boxes = box_prompts
            self.has_boxes = True
        elif box_prompts is None:
            self.boxes = None
            self.has_boxes = False
        
    def set_masks(self, mask_prompts: dict[int, np.ndarray] | np.ndarray | None):
        assert isinstance(mask_prompts, (dict, np.ndarray, type(None))), 'Mask prompts must be given either as a numpy array for 3d images, or as a dict of z_slice: slice_mask for 2d images'
        self.masks = mask_prompts
        self.has_masks = True

    def get_dict_for_2d(self):
        """
        Serialises prompts into a dictionary of z_slice: prompts for that slice. Appropriate for 2d models, not so much for 3d models.
        """
        prompts_dict = {}
        for slice_idx in self.get_slices_to_infer():  # Only useful if the prompts are supplied in the or
            slice_box = self.boxes[slice_idx] if slice_idx in self.boxes.keys() else None

            slice_coords_mask = self.coords[:, 2] == slice_idx
            slice_coords, slice_labs = (
                self.coords[slice_coords_mask, :2],
                self.labels[slice_coords_mask],
            )  # Leave out z coordinate in slice_coords

            slice_mask = self.masks[slice_idx] if slice_idx in self.masks.keys() else None

            prompts_dict[slice_idx] = {"box": slice_box, "points": (slice_coords, slice_labs), "mask": slice_mask}

        return prompts_dict

    def get_dof(self):
        """Get the degress of freedom of the current prompt step."""
        return 0 # Temporary implementation - may change to something more substantial later.

    def __getitem__(self, index):
        return self.prompts_dict[index]

    def __setitem__(self, index, value):
        self.prompts_dict[index] = value

    def get_slices_to_infer(
        self,
    ) -> np.ndarray:
        points_zs = set()
        if self.coords is not None:
            points_zs = set(self.coords[:, 0])
            # points_zs = set(self.coords[:, 2])
        boxes_zs = set()
        if self.boxes is not None:
            boxes_zs = set(self.boxes.keys())
        slices_to_infer = points_zs.union(boxes_zs)
        slices_to_infer = np.sort(list(slices_to_infer)) # Ensure the slices to infer are given in ascending order
        return slices_to_infer


def _merge_dict_prompts(prompt_dict_1: dict[int, np.ndarray] | None, prompt_dict_2: dict[int, np.ndarray] | None) -> dict[int, np.ndarray] | None:
    """
    Merges prompts given as a dictionary, ensuring that they don't have any conflicts (to merge, any slice indices they share must have the same prompt)
    """
    if prompt_dict_1 is None:
        return prompt_dict_2
    if prompt_dict_2 is None:
        return prompt_dict_1
    
    merged_dict = prompt_dict_1.copy()
    for slice_idx, prompt in prompt_dict_1.items():
        if slice_idx in prompt_dict_2.keys() and not np.array_equal(prompt_dict_1[slice_idx], prompt_dict_2[slice_idx]):
            raise ValueError("Merging would cause having two distinct box/mask prompts on one slice, which is not permitted")
    merged_dict.update(prompt_dict_2)
    
    return merged_dict


def _merge_point_prompts(coords1: np.ndarray | None, labels1: np.ndarray | None, coords2: np.ndarray | None, labels2: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Merges two point prompts, handling if either/both is/are None.
    """
    if coords1 is None:
        return coords2, labels2
    if coords2 is None:
        return coords1, labels1
    
    coords1 = np.atleast_2d(coords1)
    coords2 = np.atleast_2d(coords2)

    coords = np.concatenate([coords1, coords2], axis=0)
    labels = np.concatenate([labels1, labels2])

    return coords, labels


def _merge_two_sparse_promptsteps(prompt1: PromptStep, prompt2: PromptStep) -> PromptStep:
    """
    Merges two promptsteps. Requires that if prompt 1 and prompt 2 have box prompts for the same slice, then the box prompts must be the same
    """

    coords, labels = _merge_point_prompts(prompt1.coords, prompt1.labels, prompt2.coords, prompt2.labels)   
    boxes = _merge_dict_prompts(prompt1.boxes, prompt2.boxes)

    merged_prompt = PromptStep(point_prompts=(coords, labels), box_prompts=boxes)

    return merged_prompt


def merge_sparse_prompt_steps(prompt_steps: list[PromptStep]) -> PromptStep:
    """
    Merge a list of prompt steps into one single prompt step
    """
    assert isinstance(prompt_steps, list), 'prompt_steps argument must be a list'
    assert len(prompt_steps) > 0, 'At least one prompt step must be passed'
    assert isinstance(prompt_steps[0], PromptStep), f'Expected elements of prompt_steps to be of type PromptStep, got {type(prompt_steps[0])}'
    merged_prompt_step = prompt_steps[0]

    for ps in prompt_steps[1:]:
        assert isinstance(ps, PromptStep), f'Expected elements of prompt_steps to be of type PromptStep, got {type(prompt_steps[0])}'
        merged_prompt_step = _merge_two_sparse_promptsteps(merged_prompt_step, ps)

    return merged_prompt_step
