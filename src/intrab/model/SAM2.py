from copy import deepcopy

from loguru import logger
from intrab.model.SAM import SAMInferer
from intrab.prompts.prompt import PromptStep

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    sam2 = None

import nibabel as nib
import logging

import torch

from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_hf(model_id, **kwargs):

    from huggingface_hub import hf_hub_download

    model_id_to_filenames = {
        "facebook/sam2-hiera-tiny": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
        "facebook/sam2-hiera-small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
        "facebook/sam2-hiera-base-plus": (
            "sam2_hiera_b+.yaml",
            "sam2_hiera_base_plus.pt",
        ),
        "facebook/sam2-hiera-large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
    }
    config_name, checkpoint_name = model_id_to_filenames[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


class SAM2Inferer(SAMInferer):

    def load_model(self, checkpoint_path, device):
        sam2_model = build_sam2_hf("facebook/sam2-hiera-large")
        # SAM1 values:
        # pixel_mean=[123.675, 116.28, 103.53],
        # pixel_std = ([58.395, 57.12, 57.375],)

        sam2_model.pixel_mean = 0
        sam2_model.pixel_std = 1
        sam2_model.image_encoder.img_size = sam2_model.image_size

        return sam2_model

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
                    image_embedding = self.model.image_encoder(slice.to(self.device))["vision_features"]
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
