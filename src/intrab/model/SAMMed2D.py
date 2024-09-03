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
from nibabel.orientations import io_orientation, ornt_transform
from loguru import logger

from intrab.utils.SAMMed2D_segment_anything import sam_model_registry as registry_sammed2d
from intrab.prompts.prompt import PromptStep
from intrab.model.inferer import Inferer


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

    def set_image_old(self, img_path):
        if self._image_already_loaded(img_path=img_path):
            return
        if self.image_embeddings_dict:
            self.image_embeddings_dict = {}
        img = nib.load(img_path)
        img_ras = img  # Set in case already in RAS
        affine = img.affine

        if nib.aff2axcodes(affine) != ("R", "A", "S"):
            img_ras = nib.as_closest_canonical(img)

        ornt_old = io_orientation(img.affine)
        ornt_new = io_orientation(img_ras.affine)
        ornt_trans = ornt_transform(ornt_new, ornt_old)
        img_data = img_ras.get_fdata()
        img_data = img_data.transpose(2, 1, 0)  # Reorient to zyx

        def inv_trans(seg: np.array):
            seg = seg.transpose(2, 1, 0)  # Reorient back from zyx to RAS
            seg_nib = nib.Nifti1Image(seg, img.affine)
            seg_orig_ori = seg_nib.as_reoriented(ornt_trans)

            return seg_orig_ori

        self.img, self.inv_trans = img_data, inv_trans
        self.image_set = True

    def set_image(self, img_path: Path):
        if self._image_already_loaded(img_path=img_path):
            return
        # Load in and reorient to RAS
        if self.image_embeddings_dict:
            self.image_embeddings_dict = {}

        self.img, self.inv_trans = self.transform_to_model_coords(img_path, None)
        self.loaded_image = img_path

    def transform_to_model_coords(self, nifti: Path | nib.Nifti1Image, is_seg: bool) -> np.ndarray:
        if isinstance(nifti, (str, Path)):
            nifti: nib.Nifti1Image = nib.load(nifti)
        orientation_old = io_orientation(nifti.affine)

        if nib.aff2axcodes(nifti.affine) != ("R", "A", "S"):
            nifti = nib.as_closest_canonical(nifti)
        orientation_new = io_orientation(nifti.affine)
        orientation_transform = ornt_transform(orientation_new, orientation_old)
        data = nifti.get_fdata()
        data = data.transpose(2, 1, 0)  # Reorient to zyx

        def inv_trans(arr: np.ndarray):
            arr = arr.transpose(2, 1, 0)
            arr_nib = nib.Nifti1Image(arr, nifti.affine)
            arr_orig_ori = arr_nib.as_reoriented(orientation_transform)
            return arr_orig_ori

        # Return the data in the new format and transformation function
        return data, inv_trans

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

    def clear_embeddings(self):
        self.image_embeddings_dict = {}

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
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)

        return coords

    def apply_boxes(self, boxes, original_size, new_size):  # Copied over from SAM-Med2D predictor_sammed.py
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
        return boxes.reshape(-1, 4)

    def preprocess_img(self, img, slices_to_process):
        slices_processed = {}
        for slice_idx in slices_to_process:
            slice = img[slice_idx, ...]

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

    def preprocess_prompt(self, prompt):
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
                slice_coords_mask = coords_resized[:, 2] == slice_idx
                slice_coords, slice_labs = (
                    coords_resized[slice_coords_mask, :2],
                    labs[slice_coords_mask],
                )  # Leave out z coordinate in slice_coords
                slice_coords, slice_labs = slice_coords.unsqueeze(0).to(self.device), slice_labs.unsqueeze(0).to(
                    self.device
                )  # add batch dimension, move to device.
                preprocessed_prompts_dict[slice_idx]["point"] = (slice_coords, slice_labs)

        if prompt.has_boxes:
            for slice_index, box in prompt.boxes.items():
                box = np.array(box)
                box = self.apply_boxes(box, (self.H, self.W), self.new_size)
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
    def predict(self, prompt, mask_dict={}, return_logits=False, transform=True):
        if not (isinstance(prompt, PromptStep)):
            raise TypeError(f"Prompts must be supplied as an instance of the Prompt class.")
        if prompt.has_boxes and prompt.has_points:
            logger.warning("Both point and box prompts have been supplied; the model has not been trained on this.")
        slices_to_infer = prompt.get_slices_to_infer()

        if self.loaded_image is None:
            raise RuntimeError("Need to set an image to predict on!")

        prompt = deepcopy(prompt)

        self.D, self.H, self.W = self.img.shape
        self.original_size = (self.H, self.W)

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

        # Reorient to original orientation and return with metadata
        # Turn into Nifti object in original space
        segmentation = self.inv_trans(segmentation)

        return segmentation, low_res_logits
