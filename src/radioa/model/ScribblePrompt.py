from pathlib import Path
from typing import Sequence
import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from argparse import Namespace
import nibabel as nib
from loguru import logger
from scribbleprompt import ScribblePromptUNet
from radioa.prompts.prompt import PromptStep
from radioa.model.inferer import Inferer
from radioa.utils.transforms import orig_to_SAR_dense, orig_to_canonical_sparse_coords
from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib


class ScribblePromptInferer(Inferer):
    pass_prev_prompts = True  # Flag to track whether in interactive steps previous prompts should be passed, or only the mask and the new prompt
    dim = 2
    supported_prompts: Sequence[str] = ("box", "point", "mask")
    transform_reverses_order = True

    def __init__(self, checkpoint_path, device):
        image_size = 128
        self.new_size = (image_size, image_size)
        super().__init__(checkpoint_path, device)
        self.logit_threshold = 0  # Hardcoded

        self.image_embeddings_dict = {}

    def load_model(self, checkpoint_path, device):
        return ScribblePromptUNet(device=device)

    def set_image(self, img_path: str | Path) -> None:
        if self._image_already_loaded(img_path=img_path):
            return
        self.image_embeddings_dict = {}
        img_nib = load_any_to_nib(img_path)
        self.orig_affine = img_nib.affine
        self.orig_shape = img_nib.shape

        self.img, self.inv_trans_dense = self.transform_to_model_coords_dense(img_nib, is_seg=False)
        self.new_shape = self.img.shape
        self.loaded_image = img_path

    def transform_to_model_coords_dense(self, nifti: str | Path | nib.Nifti1Image, is_seg: bool) -> np.ndarray:
        # Model space is always throughplane first (commonly the z-axis)
        data, inv_trans = orig_to_SAR_dense(nifti)

        return data, inv_trans

    def transform_to_model_coords_sparse(self, coords: np.ndarray) -> np.ndarray:
        return orig_to_canonical_sparse_coords(coords, self.orig_affine, self.orig_shape)

    def segment(self, image, points, box, mask):
        if points is None:
            points = (None, None)

        new_mask = self.model.predict(
            image[None,None],        # (B, 1, H, W)
            points[0], # (B, n, 2)
            points[1], # (B, n)
            None,    # (B, 2, H, W)
            box,          # (B, n, 4)
            mask,   # (B, 1, H, W)
        )

        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(image.cpu().numpy(), name="Image")
        # # p = points[0][0].cpu().numpy()
        # # viewer.add_points([[p[0,1], p[0,0]]], name="Points")
        # box_mask = np.zeros_like(image.cpu().numpy())
        # box_mask[box[0,0,1].item():box[0,0,3].item(), box[0,0,0].item():box[0,0,2].item()] = 1
        # viewer.add_labels(box_mask.astype(np.uint8), name="Box")
        # viewer.add_labels((new_mask[:,0] > 0.5).cpu().numpy().astype(np.uint8), name="Mask")
        # napari.run()
        return new_mask


    def transforms(self, new_size):  # Copied over from SAM-Med2D predictor_sammed.py
        Transforms = []
        new_h, new_w = new_size
        Transforms.append(
            A.Resize(int(new_h), int(new_w), interpolation=cv2.INTER_NEAREST)
        )  # note nearest neighbour interpolation.
        Transforms.append(ToTensorV2(p=1.0))
        return A.Compose(Transforms, p=1.0)

    def apply_coords(self, coords, original_size, new_size):  # Copied over from SAM-Med2D predictor_sammed.py
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = deepcopy(coords).astype(float)
        coords[..., 2] = coords[..., 2] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)

        return coords

    def apply_boxes(self, boxes, original_size, new_size):  # Copied over from SAM-Med2D predictor_sammed.py
        boxes = self.apply_coords(boxes.reshape(2, 3), original_size, new_size)
        boxes = boxes[:, 1:]  # Remove z coord
        return boxes.reshape(-1, 4)

    def preprocess_img(self, img, slices_to_process):
        slices_processed = {}
        for slice_idx in slices_to_process:
            slice = img[slice_idx, ...].astype(np.float32)
            # normalize to 0-1
            slice = torch.from_numpy((slice - slice.min()) / (slice.max() - slice.min())).to(self.device)
            slice = F.interpolate(slice[None,None], size=self.new_size, mode='bilinear')[0,0]
            slices_processed[slice_idx] = slice

        return slices_processed


    def preprocess_prompt(self, prompt, promptstep_in_model_coord_system=False):
        """
        Preprocessing steps:
            - Modify in line with the volume cropping
            - Modify in line with the interpolation
            - Collect into a dictionary of slice:slice prompt
        """

        preprocessed_prompts_dict = {
            slice_idx: {"point": None, "box": None} for slice_idx in prompt.get_slices_to_infer()
        }

        if prompt.has_points:
            coords, labs = prompt.coords, prompt.labels
            coords, labs = np.array(coords).astype(float), np.array(labs).astype(int)

            # coords = coords[:,[0,2,1]] # Change from ZYX to ZXY
            coords_resized = self.apply_coords(coords, (self.H, self.W), self.new_size)
            #coords_resized = np.int16(coords_resized)

            # Convert to torch tensor
            coords_resized = torch.as_tensor(coords_resized, dtype=torch.int16)
            labs = torch.as_tensor(labs, dtype=int)

            # Collate
            for slice_idx in prompt.get_slices_to_infer():
                slice_coords_mask = coords_resized[:, 0] == slice_idx
                slice_coords, slice_labs = (  # Subset to slice
                    coords_resized[slice_coords_mask],
                    labs[slice_coords_mask],
                )
                slice_coords = slice_coords[:, [2, 1]]  # leave out z and reorder
                slice_coords, slice_labs = slice_coords.unsqueeze(0).to(self.device), slice_labs.unsqueeze(0).to(
                    self.device
                )  # add batch dimension, move to device.
                preprocessed_prompts_dict[slice_idx]["point"] = (slice_coords, slice_labs)

        if prompt.has_boxes:
            for slice_index, box in prompt.boxes.items():
                box = np.array(
                    [slice_index, box[1], box[0], slice_index, box[3], box[2]]
                )  # Transform reverses coordinates, so desired points must be given as zyx
                box = self.apply_boxes(box, (self.H, self.W), self.new_size)
                box = np.array([box[0, 1], box[0, 0], box[0, 3], box[0, 2]])[None]  # Desperate fix attempt
                box = np.round(box)
                # box = box[:, :, [1, 0, 3, 2]] #
                box = torch.as_tensor(box, dtype=torch.int16, device=self.device)
                box = box[None, :]
                preprocessed_prompts_dict[slice_index]["box"] = box.to(self.device)

        return preprocessed_prompts_dict

    def postprocess_slices(self, slice_mask_dict, return_logits):
        """
        Postprocessing steps:
            - Combine inferred slices into one volume, interpolating back to the original volume size
            - Turn logits into binary mask
            - Invert crop/pad to get back to original image dimensions
        """
        # Combine segmented slices into a volume with 0s for non-segmented slices

        dtype = np.float32 if return_logits else np.uint8
        segmentation = np.zeros((self.D, self.H, self.W), dtype)

        for z, low_res_mask in slice_mask_dict.items():
            mask = F.interpolate(low_res_mask, self.original_size, mode="bilinear")

            if not return_logits:
                mask = (mask > 0.5).to(torch.uint8)

            segmentation[z, :, :] = mask.cpu().numpy()

        return segmentation

    @torch.no_grad()
    def predict(
            self, prompt, return_logits=False, prev_seg=None, promptstep_in_model_coord_system=False
    ) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
        if not (isinstance(prompt, PromptStep)):
            raise TypeError(f"Prompts must be supplied as an instance of the Prompt class.")
        if prompt.has_boxes and prompt.has_points:
            logger.warning("Both point and box prompts have been supplied; the model has not been trained on this.")

        if self.loaded_image is None:
            raise RuntimeError("Need to set an image to predict on!")

        prompt = deepcopy(prompt)
        if not promptstep_in_model_coord_system:
            prompt = self.transform_promptstep_to_model_coords(prompt)
        slices_to_infer = prompt.get_slices_to_infer()

        # Plotting
        # coords_memory = [[x[0],x[1],x[2]] for x in prompt.coords]

        self.D, self.H, self.W = self.img.shape
        self.original_size = (self.H, self.W)

        mask_dict = prompt.masks if prompt.masks is not None else {}
        preprocessed_prompt_dict = self.preprocess_prompt(prompt)
        slices_to_process = [
            slice_idx for slice_idx in slices_to_infer if slice_idx not in self.image_embeddings_dict.keys()
        ]

        slices_processed = self.preprocess_img(self.img, slices_to_process)

        self.slice_lowres_outputs = {}
        for slice_idx in slices_to_infer:
            slice = slices_processed[slice_idx]

            # Get prompts
            slice_points, slice_box = (
                preprocessed_prompt_dict[slice_idx]["point"],
                preprocessed_prompt_dict[slice_idx]["box"],
            )
            slice_mask = (
                torch.from_numpy(mask_dict[slice_idx]).to(self.device).unsqueeze(0).unsqueeze(0)
                if slice_idx in mask_dict.keys()
                else None
            )

            # Infer
            slice_raw_outputs = self.segment(
                image=slice, points=slice_points, box=slice_box, mask=slice_mask,
            )
            self.slice_lowres_outputs[slice_idx] = slice_raw_outputs

        low_res_logits = {k: torch.sigmoid(v).squeeze().cpu().numpy() for k, v in self.slice_lowres_outputs.items()}

        segmentation = self.postprocess_slices(self.slice_lowres_outputs, return_logits)

        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(self.img, name='img')
        # viewer.add_labels(segmentation, name='seg')
        # # viewer.add_points(coords_memory, size=4, name='prompt')
        # box = np.zeros_like(self.img).astype(np.uint8)
        # for slice_idx in slices_to_infer:
        #     box[slice_idx, prompt.boxes[slice_idx][1]:prompt.boxes[slice_idx][3], prompt.boxes[slice_idx][0]:prompt.boxes[slice_idx][2]] = 1
        # viewer.add_labels(box, name='box')
        # napari.run()

        # Fill in missing slices using a previous segmentation if desired
        if prev_seg is not None:
            segmentation = self.merge_seg_with_prev_seg(segmentation, prev_seg, slices_to_infer)

        # Reorient to original orientation and return with metadata
        # Turn into Nifti object in original space
        # Turn into Nifti object in original space
        segmentation_model_arr = segmentation
        segmentation_orig_nib = self.inv_trans_dense(segmentation)

        return segmentation_orig_nib, low_res_logits, segmentation_model_arr
