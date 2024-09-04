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
            points_zs = set(self.coords[:, 2])
        boxes_zs = set()
        if self.boxes is not None:
            boxes_zs = set(self.boxes.keys())
        slices_to_infer = points_zs.union(boxes_zs)
        slices_to_infer = np.sort(list(slices_to_infer)) # Ensure the slices to infer are given in ascending order
        return slices_to_infer


def _merge_two_promptsteps(prompt1: PromptStep, prompt2: PromptStep) -> PromptStep:
    # Need to have empty values as {}, not None. Need to discuss with Tilo since he preferred None
    if prompt1.boxes is None: prompt1.boxes = {}
    if prompt2.boxes is None: prompt2.boxes = {}
    if prompt1.masks is None: prompt1.masks = {}
    if prompt2.masks is None: prompt2.masks = {}

    # Merge boxes
    boxes = prompt1.boxes
    for slice_idx, bbox in prompt1.boxes.items():  # Check no slice gets two distinct boxes
        if slice_idx in prompt2.boxes.keys() and not np.array_equal(prompt1.boxes[slice_idx], prompt2.boxes[slice_idx]):
            raise ValueError("Merging would cause having two distinct boxes on one slice, which is not permitted")
        boxes.update(prompt2.boxes)

    # Merge points
    # Ensure both point arrays are 2d to permit concatenation
    # This will break if the coords are none.
    coords1 = np.atleast_2d(prompt1.coords)
    coords2 = np.atleast_2d(prompt2.coords)

    coords = np.concatenate([coords1, coords2], axis=0)
    labels = np.concatenate([prompt1.labels, prompt2.labels])

    
    # Merge masks 
    masks = prompt1.masks
    for slice_idx, mask in prompt1.masks.items():  # Check no slice gets two distinct boxes
        if slice_idx in prompt2.masks.keys() and not np.array_equal(prompt1.masks[slice_idx], prompt2.masks[slice_idx]):
            raise ValueError("Merging would cause having two distinct masks on one slice, which is not permitted")
        masks.update(prompt2.masks)

    merged_prompt = PromptStep(point_prompts=(coords, labels), box_prompts=boxes, mask_prompts=masks)

    return merged_prompt

def merge_prompt_steps(prompt_steps: list[PromptStep]) -> PromptStep:
    """
    Merge a list of prompt steps into one singple prompt step
    """
    assert isinstance(prompt_steps, list), 'prompt_steps argument must be a list'
    assert len(prompt_steps) > 0, 'At least one prompt step must be passed'
    assert isinstance(prompt_steps[0], PromptStep), f'Expected elements of prompt_steps to be of type PromptStep, got {type(prompt_steps[0])}'
    merged_prompt_step = prompt_steps[0]

    for ps in prompt_steps[1:]:
        assert isinstance(ps, PromptStep), f'Expected elements of prompt_steps to be of type PromptStep, got {type(prompt_steps[0])}'
        merged_prompt_step = _merge_two_promptsteps(merged_prompt_step, ps)

    return merged_prompt_step
