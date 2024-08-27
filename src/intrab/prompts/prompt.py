import numpy as np


class Points:
    def __init__(self, coords, labels):
        self.coords = np.array(coords)
        self.labels = np.array(labels)

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
        point_prompts: Points | tuple[np.ndarray, np.ndarray] = None,
        box_prompts: Boxes | dict[int, np.ndarray] = None,
    ):
        """
        Initialise either empty, with points or with boxes.
        Points must be supplied as a pair (coords, labels), with coords nx3 containing the coordinates (xyz) of the prompts, or as a Points object
        Boxes must be supplied as a dictionary of z_slice: (x_min, y_min, x_max, y_max) or as a Boxes object
        """
        # Default initiation
        self.coords = np.empty((0, 3))
        self.labels = np.empty(0)

        self.boxes = {}
        self.has_points = self.has_boxes = False

        # Handle point initiation
        if isinstance(point_prompts, Points):
            self.coords, self.labels = point_prompts.coords, point_prompts.labels
            self.has_points = True

        elif point_prompts is not None:
            self.coords, self.labels = np.array(point_prompts[0]), np.array(point_prompts[1])
            self.has_points = True

        # Handle box initiation
        if isinstance(box_prompts, Boxes):
            self.boxes = self.boxes.value
            self.has_boxes = True
        elif box_prompts is not None:
            self.boxes = box_prompts
            self.has_boxes = True

        # Process into dictionary
        self.slices_to_infer = self.get_slices_to_infer(self.coords, self.boxes)

    def get_dict(self):
        self.prompts_dict = {}
        for slice_idx in self.slices_to_infer:  # Only useful if the prompts are supplied in the or
            slice_box = self.boxes[slice_idx] if slice_idx in self.boxes.keys() else None

            slice_coords_mask = self.coords[:, 2] == slice_idx
            slice_coords, slice_labs = (
                self.coords[slice_coords_mask, :2],
                self.labels[slice_coords_mask],
            )  # Leave out z coordinate in slice_coords

            self.prompts_dict[slice_idx] = {"box": slice_box, "points": (slice_coords, slice_labs)}

    def __getitem__(self, index):
        return self.prompts_dict[index]

    def __setitem__(self, index, value):
        self.prompts_dict[index] = value

    @staticmethod
    def get_slices_to_infer(coords, boxes):
        points_zs = set(coords[:, 2])
        boxes_zs = set(boxes.keys())
        slices_to_infer = points_zs.union(boxes_zs)
        return slices_to_infer
