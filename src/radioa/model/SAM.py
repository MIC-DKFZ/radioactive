from pathlib import Path
import nrrd
import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from argparse import Namespace
import nibabel as nib
from loguru import logger
from radioa.model.inferer import Inferer
from radioa.prompts.prompt import PromptStep
from radioa.utils.SAMMed3D_segment_anything.build_sam import sam_model_registry as registry_sam
from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib
from radioa.utils.transforms import (
    ResizeLongestSide,
    orig_to_SAR_dense,
    orig_to_canonical_sparse_coords,
    transform_prompt_to_model_coords,
)


def load_sam(checkpoint_path, device="cuda", image_size=1024):
    args = Namespace()
    args.image_size = image_size
    args.sam_checkpoint = checkpoint_path
    args.model_type = "vit_h"
    model = registry_sam[args.model_type](args).to(device)
    model.eval()
    return model


class SAMInferer(Inferer):
    pass_prev_prompts = True  # In supplied demos, sam doesn't take previous prompts, but this vastly increases performance when the model greatly oversegments, for example.
    dim = 2
    supported_prompts = ("box", "point", "mask")
    transform_reverses_order = True

    def __init__(self, checkpoint_path, device):
        super(SAMInferer, self).__init__(checkpoint_path, device)
        self.prev_mask = None
        self.target_volume_shape = 128  # Hardcoded to match training
        self.target_slice_shape = 256  # Hardcoded to match training
        self.inputs = None
        self.mask_threshold = 0
        self.device = device
        self.image_embeddings_dict = {}
        self.verbose = True
        print("init inferer")
        self.pixel_mean = self.model.pixel_mean
        self.pixel_std = self.model.pixel_std
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)
        self.input_size = None

    def load_model(self, checkpoint_path, device):
        return load_sam(checkpoint_path, device)

    @torch.no_grad()
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
            multimask_output=True,
        )

        # Obtain the best mask (measured by predicted iou) from the 3 returned masks
        iou_predictions = iou_predictions[0]  # Indexing within batch : we have batch size 1
        max_value, max_index = torch.max(iou_predictions, dim=0)
        best_mask = low_res_masks[0, max_index]

        return best_mask

    def set_image(self, img_path: Path | str):
        img_path = Path(img_path)
        if self._image_already_loaded(img_path=img_path):
            return
        self.image_embeddings_dict = {}
        img_nib = load_any_to_nib(img_path)
        self.old_img = img_nib.get_fdata()
        self.orig_affine = img_nib.affine
        self.orig_shape = img_nib.shape
        self.img_name = img_path.stem
        self.img, self.inv_trans_dense = self.transform_to_model_coords_dense(img_nib, is_seg=False)
        self.loaded_image = img_path
        self.new_shape = self.img.shape
        self.loaded_image = img_path

        # clip image to 0.5% - 99.5%
        self.global_min = np.percentile(self.img, 0.5)
        self.global_max = np.percentile(self.img, 99.5)

        # Clip the image array
        self.img = np.clip(self.img, self.global_min, self.global_max)

    def preprocess_img(self, img, slices_to_process):
        """
        Preprocessing steps
            - Extract slice, resize (maintaining aspect ratio), pad edges
        """

        # Perform slicewise processing and collect back into a volume at the end
        slices_processed = {}
        for slice_idx in slices_to_process:
            slice = img[slice_idx, ...]  # Now HW
            slice = np.round((slice - self.global_min) / (self.global_max - self.global_min + 1e-10) * 255.0).astype(
                np.uint8
            )  # Change to 0-255 scale
            slice = np.repeat(slice[..., None], repeats=3, axis=-1)  # Add channel dimension to make it RGB-like
            slice = self.transform.apply_image(slice)
            slice = torch.as_tensor(slice, device=self.device)
            slice = slice.permute(2, 0, 1).contiguous()[
                None, :, :, :
            ]  # Change to BCHW, make memory storage contiguous.

            # if self.input_size is None:
            #     self.input_size = tuple(
            #         slice.shape[-2:]
            #     )  # Store the input size pre-padding if it hasn't been done yet
            self.input_size = tuple(slice.shape[-2:])
            slice = (slice - self.pixel_mean) / self.pixel_std

            h, w = slice.shape[-2:]
            padh = self.model.image_encoder.img_size - h
            padw = self.model.image_encoder.img_size - w
            slice = F.pad(slice, (0, padw, 0, padh))

            slices_processed[slice_idx] = slice
        self.slices_processed = slices_processed
        return slices_processed

    def transform_to_model_coords_dense(self, nifti: Path | nib.Nifti1Image, is_seg: bool) -> np.ndarray:
        # Model space is always throughplane first (commonly the z-axis)
        data, inv_trans = orig_to_SAR_dense(nifti)

        return data, inv_trans

    def transform_to_model_coords_sparse(self, coords: np.ndarray) -> np.ndarray:
        return orig_to_canonical_sparse_coords(coords, self.orig_affine, self.orig_shape)

    def preprocess_prompt(self, prompt: PromptStep, promptstep_in_model_coord_system=False):
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
            coords = prompt.coords
            labs = prompt.labels

            coords_resized = self.transform.apply_coords(coords, (self.H, self.W))

            # Convert to torch tensor
            coords_resized = torch.as_tensor(coords_resized, dtype=torch.float)
            labs = torch.as_tensor(labs, dtype=int)

            # Collate

            for slice_idx in prompt.get_slices_to_infer():
                slice_coords_mask = coords_resized[:, 0] == slice_idx
                slice_coords, slice_labs = (  # subset to slice
                    coords_resized[slice_coords_mask],
                    labs[slice_coords_mask],
                )
                slice_coords = slice_coords[:, [2, 1]]  # leave out z and reorder
                slice_coords, slice_labs = slice_coords.unsqueeze(0), slice_labs.unsqueeze(0)
                preprocessed_prompts_dict[slice_idx]["point"] = (
                    slice_coords.to(self.device),
                    slice_labs.to(self.device),
                )

        if prompt.has_boxes:
            for slice_index, box in prompt.boxes.items():
                box = np.array([slice_index, box[1], box[0], slice_index, box[3], box[2]])
                box = self.transform.apply_boxes(box, (self.H, self.W))
                box = np.array([box[0, 1], box[0, 0], box[0, 3], box[0, 2]])[None]  # Desperate fix attempt
                box = torch.as_tensor(box, dtype=torch.float, device=self.device)
                box = box[None, :]
                preprocessed_prompts_dict[slice_index]["box"] = box.to(self.device)

        return preprocessed_prompts_dict

    def postprocess_slices(self, slice_mask_dict, return_logits):
        """
        Postprocessing steps:
            - TODO
        """
        # Combine segmented slices into a volume with 0s for non-segmented slices
        dtype = np.float32 if return_logits else np.uint8
        segmentation = np.zeros((self.D, self.H, self.W), dtype)
        for z, low_res_mask in slice_mask_dict.items():
            low_res_mask = low_res_mask.unsqueeze(0).unsqueeze(0)  # Include batch and channel dimensions
            mask_input_res = F.interpolate(
                low_res_mask,
                (self.model.image_encoder.img_size, self.model.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )  # upscale low res mask to mask as in input_size
            mask_input_res = mask_input_res[
                ..., : self.input_size[0], : self.input_size[1]
            ]  # Crop out any segmentations created in the padded sections
            slice_mask = F.interpolate(mask_input_res, self.original_size, mode="bilinear", align_corners=False)
            if not return_logits:
                slice_mask = (slice_mask > 0.5).to(torch.uint8)
            segmentation[z, :, :] = slice_mask.cpu().numpy()

        return segmentation

    def predict(
        self,
        prompt: PromptStep,
        return_logits: bool = False,
        prev_seg=None,
        promptstep_in_model_coord_system=False,
    ) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
        if not isinstance(prompt, PromptStep):
            raise TypeError(f"Prompts must be supplied as an instance of the Prompt class.")
        if prompt.has_boxes and prompt.has_points:
            logger.warning("Both point and box prompts have been supplied; the model has not been trained on this.")

        # Transform prompt if needed
        if not promptstep_in_model_coord_system:
            prompt = self.transform_promptstep_to_model_coords(prompt)

        slices_to_infer = prompt.get_slices_to_infer()

        prompt = deepcopy(prompt)

        self.D, self.H, self.W = self.img.shape
        self.original_size = (
            self.H,
            self.W,
        )  # Used for the transform class, which is taken from the original SAM code, hence the 2D size
        # print(self.img.shape)
        mask_dict = prompt.masks if prompt.masks is not None else {}
        preprocessed_prompt_dict = self.preprocess_prompt(prompt)
        slices_to_process = [
            slice_idx for slice_idx in slices_to_infer if slice_idx not in self.image_embeddings_dict.keys()
        ]

        slices_processed = self.preprocess_img(self.img, slices_to_process)

        self.slice_lowres_outputs = {}

        for slice_idx in slices_to_infer:
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
            )  # Add batch dimensions
            self.slice_lowres_outputs[slice_idx] = slice_raw_outputs

        low_res_logits = {k: torch.sigmoid(v).squeeze().cpu().numpy() for k, v in self.slice_lowres_outputs.items()}
        segmentation = self.postprocess_slices(self.slice_lowres_outputs, return_logits)

        # Fill in missing slices using a previous segmentation if desired
        if prev_seg is not None:
            segmentation = self.merge_seg_with_prev_seg(segmentation, prev_seg, slices_to_infer)

        # Reorient to original orientation and return with metadata
        # Turn into Nifti object in original space
        segmentation_model_arr = segmentation
        segmentation_orig = self.inv_trans_dense(segmentation)
        return segmentation_orig, low_res_logits, segmentation_model_arr
