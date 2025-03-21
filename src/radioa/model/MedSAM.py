from pathlib import Path
from loguru import logger
import torch
import numpy as np
import torch.nn.functional as F
import cv2
import nibabel as nib


from radioa.model.inferer import Inferer
from radioa.prompts.prompt import PromptStep
from radioa.utils.MedSAM_segment_anything import sam_model_registry as registry_medsam
from radioa.utils.transforms import orig_to_SAR_dense, orig_to_canonical_sparse_coords
from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib


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

    def transform_to_model_coords_dense(self, nifti: Path | nib.Nifti1Image, is_seg: bool) -> np.ndarray:
        data, inv_trans = orig_to_SAR_dense(nifti)

        return data, inv_trans

    def transform_to_model_coords_sparse(self, coords: np.ndarray) -> np.ndarray:
        return orig_to_canonical_sparse_coords(coords, self.orig_affine, self.orig_shape)

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
            lower_bound, upper_bound = np.percentile(slice[slice > 0], 0.5), np.percentile(slice[slice > 0], 99.5)
            slice = np.clip(slice, lower_bound, upper_bound)

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

    def predict(
        self, prompt: PromptStep, prev_seg=None, promptstep_in_model_coord_system: bool = False
    ) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
        if not (isinstance(prompt, PromptStep)):
            raise TypeError(f"Prompts must be supplied as an instance of the Prompt class.")
        if prompt.has_points:
            prompt.points = None
            logger.warning("MedSAM does not support point prompts. Ignoring points.")

        if not promptstep_in_model_coord_system:
            prompt = self.transform_promptstep_to_model_coords(prompt)

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

        # Fill in missing slices using a previous segmentation if desired
        if prev_seg is not None:
            segmentation = self.merge_seg_with_prev_seg(segmentation, prev_seg, slices_to_infer)

        # Turn into Nifti object in original space
        segmentation_model_arr = segmentation
        segmentation_orig_nib = self.inv_trans_dense(segmentation)

        return segmentation_orig_nib, low_res_logits, segmentation_model_arr
