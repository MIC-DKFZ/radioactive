from pathlib import Path
from loguru import logger
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import cv2
import nibabel as nib
from nibabel.orientations import io_orientation, ornt_transform

from intrab.model.inferer import Inferer
from intrab.prompts.prompt import PromptStep
from intrab.utils.MedSAM_segment_anything import sam_model_registry as registry_medsam


def load_medsam(checkpoint_path, device="cuda"):
    medsam_model = registry_medsam["vit_b"](checkpoint=checkpoint_path)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    return medsam_model


class MedSAMInferer(Inferer):
    dim = 2
    supported_prompts = ("box",)

    def __init__(self, checkpoint_path, device):
        super(MedSAMInferer, self).__init__(checkpoint_path, device)
        self.logit_threshold = 0.5
        self.verbose = True

        self.D, self.H, self.W = None, None, None

    def load_model(self, checkpoint_path, device):
        return load_medsam(checkpoint_path, device)

    def _set_image_old(self, img_path: Path):
        if self._image_already_loaded(img_path=img_path):
            return
        # Load in and reorient to RAS
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
        if isinstance(nifti, Path):
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

    def postprocess_mask(self, mask):
        pass

    @torch.no_grad()
    def segment(self, points, box, mask, image_embedding):
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=box,
            masks=mask,
        )

        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        return low_res_pred

    def preprocess_img(self, img, slices_to_process):
        slices_processed = {}
        for slice_idx in slices_to_process:
            slice = img[slice_idx, ...]
            slice = np.repeat(
                slice[..., np.newaxis], repeats=3, axis=2
            )  # Repeat three times along a new final axis to simulate being a color image.

            slice = cv2.resize(slice, (1024, 1024), interpolation=cv2.INTER_CUBIC)

            slice = (slice - slice.min()) / np.clip(
                slice.max() - slice.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1]
            slice = slice.transpose(2, 0, 1)[None]  # HWC -> NCHW

            slices_processed[slice_idx] = torch.from_numpy(slice).float()

        return slices_processed

    def preprocess_prompt(self, prompt):
        preprocessed_prompts_dict = {
            slice_idx: {"point": None, "box": None} for slice_idx in prompt.get_slices_to_infer()
        }

        if prompt.has_boxes:
            for slice_idx, box in prompt.boxes.items():
                box_1024 = box / np.array((self.W, self.H, self.W, self.H)) * 1024
                box_torch = (
                    torch.from_numpy(box_1024).float().unsqueeze(0).unsqueeze(0)
                )  # Add 'number of boxes' and batch dimensions
                preprocessed_prompts_dict[slice_idx]["box"] = box_torch.to(self.device)

            return preprocessed_prompts_dict

    def postprocess_slices(self, slice_mask_dict):
        """
        Postprocessing steps:
            - Combine inferred slices into one volume, interpolating back to the original volume size
            - Turn logits into binary mask
        """
        # Combine segmented slices into a volume with 0s for non-segmented slices
        segmentation = torch.zeros((self.D, self.H, self.W))
        for z, low_res_mask in slice_mask_dict.items():

            low_res_mask = F.interpolate(
                low_res_mask,
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            )  # (1, 1, gt.shape)
            segmentation[z, :, :] = low_res_mask

        segmentation = (segmentation > self.logit_threshold).numpy()
        segmentation = segmentation.astype(np.uint8)

        return segmentation

    def predict(self, prompt: PromptStep, transform=True):
        if not (isinstance(prompt, PromptStep)):
            raise TypeError(f"Prompts must be supplied as an instance of the Prompt class.")
        if prompt.has_points:
            prompt.points = None
            logger.warning("MedSAM does not support point prompts. Ignoring points.")

        self.D, self.H, self.W = self.img.shape

        if self.loaded_image is None:
            raise RuntimeError("Need to set an image to predict on!")

        slices_to_infer = prompt.get_slices_to_infer()

        preprocessed_prompt_dict = self.preprocess_prompt(prompt)

        slices_to_process = [
            slice_idx for slice_idx in slices_to_infer if slice_idx not in self.image_embeddings_dict.keys()
        ]
        slices_processed = self.preprocess_img(self.img, slices_to_process)

        slice_lowres_outputs: dict[int, torch.Tensor | np.ndarray] = {}
        for slice_idx in slices_to_infer:
            if slice_idx in self.image_embeddings_dict.keys():
                image_embedding = self.image_embeddings_dict[slice_idx].to(self.device)
            else:
                slice = slices_processed[slice_idx]
                with torch.no_grad():
                    image_embedding = self.model.image_encoder(slice.to(self.device))
                self.image_embeddings_dict[slice_idx] = image_embedding.cpu()

            # Get prompts
            slice_box = preprocessed_prompt_dict[slice_idx]["box"]

            # Infer
            slice_raw_outputs = self.segment(points=None, box=slice_box, mask=None, image_embedding=image_embedding)
            slice_lowres_outputs[slice_idx] = slice_raw_outputs

        low_res_logits = {k: torch.sigmoid(v).squeeze().cpu().numpy() for k, v in slice_lowres_outputs.items()}

        segmentation = self.postprocess_slices(slice_lowres_outputs)

        # Turn into Nifti object in original space
        segmentation = self.inv_trans(segmentation)

        return segmentation, low_res_logits
