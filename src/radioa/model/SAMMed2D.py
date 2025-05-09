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

from radioa.utils.SAMMed2D_segment_anything import sam_model_registry as registry_sammed2d
from radioa.prompts.prompt import PromptStep
from radioa.model.inferer import Inferer
from radioa.utils.transforms import orig_to_SAR_dense, orig_to_canonical_sparse_coords
from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib


def load_sammed2d(checkpoint_path, image_size, device="cuda"):
    args = Namespace()
    args.image_size = image_size
    args.encoder_adapter = True
    args.sam_checkpoint = checkpoint_path
    model = registry_sammed2d["vit_b"](args).to(device)
    model.eval()

    return model


class SAMMed2DInferer(Inferer):
    pass_prev_prompts = True  # Flag to track whether in interactive steps previous prompts should be passed, or only the mask and the new prompt
    dim = 2
    supported_prompts: Sequence[str] = ("box", "point", "mask")
    transform_reverses_order = True

    def __init__(self, checkpoint_path, device):
        image_size = 256
        self.new_size = (image_size, image_size)
        super().__init__(checkpoint_path, device)
        self.logit_threshold = 0  # Hardcoded

        self.image_embeddings_dict = {}
        self.multimask_output = True  # Hardcoded to match defaults from original

        self.pixel_mean, self.pixel_std = (
            self.model.pixel_mean.squeeze().cpu().numpy(),
            self.model.pixel_std.squeeze().cpu().numpy(),
        )

    def load_model(self, checkpoint_path, device):
        return load_sammed2d(checkpoint_path, self.new_size[0], device)

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

    def segment(self, points, box, mask, image_embedding):
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=box,
            masks=mask,
        )

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
        )

        if self.multimask_output:
            max_values, max_indexs = torch.max(iou_predictions, dim=1)
            max_values = max_values.unsqueeze(1)
            iou_predictions = max_values
            low_res_masks = low_res_masks[:, max_indexs]

        return low_res_masks


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
            slice = img[slice_idx, ...]
            lower_bound, upper_bound = np.percentile(slice[slice > 0], 0.5), np.percentile(slice[slice > 0], 99.5)
            slice = np.clip(slice, lower_bound, upper_bound)

            slice = np.round((slice - slice.min()) / (slice.max() - slice.min() + 1e-6) * 255).astype(
                np.uint8
            )  # Get slice into [0,255] rgb scale
            slice = np.repeat(slice[..., None], repeats=3, axis=-1)  # Add channel dimension to make it RGB-like
            slice = (slice - self.pixel_mean) / self.pixel_std  # normalise

            transforms = self.transforms(self.new_size)
            augments = transforms(image=slice)
            slice = augments["image"][None, :, :, :]  # Add batch dimension

            slices_processed[slice_idx] = slice.float()

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

            # coords = coords[:,[2,1,0]] # Change from ZYX to XYZ
            coords_resized = self.apply_coords(coords, (self.H, self.W), self.new_size)

            # Convert to torch tensor
            coords_resized = torch.as_tensor(coords_resized, dtype=torch.float)
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
                box = torch.as_tensor(box, dtype=torch.float, device=self.device)
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
            mask = F.interpolate(low_res_mask, self.new_size, mode="bilinear", align_corners=False)
            mask = F.interpolate(
                mask, self.original_size, mode="bilinear", align_corners=False
            )  # upscale in two steps to match original code

            mask = torch.sigmoid(mask)
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
            # Get image embedding (either create it, or read it if stored and desired)
            if slice_idx in self.image_embeddings_dict.keys():
                image_embedding = self.image_embeddings_dict[slice_idx].to(self.device)
            else:
                slice = slices_processed[slice_idx]
                with torch.no_grad():
                    image_embedding = self.model.image_encoder(slice.to(self.device))
                self.image_embeddings_dict[slice_idx] = image_embedding.cpu()

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
                points=slice_points, box=slice_box, mask=slice_mask, image_embedding=image_embedding
            )
            self.slice_lowres_outputs[slice_idx] = slice_raw_outputs

        low_res_logits = {k: torch.sigmoid(v).squeeze().cpu().numpy() for k, v in self.slice_lowres_outputs.items()}

        segmentation = self.postprocess_slices(self.slice_lowres_outputs, return_logits)

        # Fill in missing slices using a previous segmentation if desired
        if prev_seg is not None:
            segmentation = self.merge_seg_with_prev_seg(segmentation, prev_seg, slices_to_infer)

        # Reorient to original orientation and return with metadata
        # Turn into Nifti object in original space
        # Turn into Nifti object in original space
        segmentation_model_arr = segmentation
        segmentation_orig_nib = self.inv_trans_dense(segmentation)

        return segmentation_orig_nib, low_res_logits, segmentation_model_arr
