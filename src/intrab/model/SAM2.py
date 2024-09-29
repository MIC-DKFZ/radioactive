from copy import deepcopy
from typing import Optional, Tuple

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

from intrab.utils.transforms import SAM2Transforms

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
    def __init__(self, checkpoint_path, device):
        super().__init__(checkpoint_path, device)
        self.resolution = self.model.image_size
        mask_threshold = 0.0 # Match sam2_image_predictor.py values
        max_hole_area = 0.0 #  Match sam2_image_predictor.py values
        max_sprinkle_area = 0.0 #  Match sam2_image_predictor.py values
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        self.multimask_output = False

    def load_model(self, checkpoint_path, device):
        sam2_model = build_sam2_hf("facebook/sam2-hiera-large")
        return sam2_model

    def preprocess_image(self, img, slices_to_process):
        # Perform slicewise processing and collect back into a volume at the end
        slices_processed = {}
        self._orig_hw = img[0].shape
        for slice_idx in slices_to_process:
            slice = img[slice_idx, ...]  # Now HW
            slice = np.round((slice - slice.min()) / (slice.max() - slice.min() + 1e-10) * 255.0).astype(
                np.uint8
            )  # Change to 0-255 scale
            slice = np.repeat(slice[..., None], repeats=3, axis=-1)  # Add channel dimension to make it RGB-like -> now HWC

            input_slice = self._transforms(slice)
        return slices_processed

    def get_features(self, preprocessed_slice:torch.Tensor) -> dict:
        backbone_out = self.model.forward_image(preprocessed_slice)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        return _features
    
    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box
    
    def preprocess_prompt(self, prompt:PromptStep) -> dict:
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
                slice_coords, slice_labs = ( # subset to slice
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
                box = np.array(
                    [slice_index, box[1], box[0], slice_index, box[3], box[2]]
                )
                box = self.transform.apply_boxes(box, (self.H, self.W))
                box = np.array([box[0, 1], box[0, 0], box[0, 3], box[0, 2]])[None]  # Desperate fix attempt
                box = torch.as_tensor(box, dtype=torch.float, device=self.device)
                box = box[None, :]
                preprocessed_prompts_dict[slice_index]["box"] = box.to(self.device)

        return preprocessed_prompts_dict
    
    @torch.no_grad()
    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using SAM2Transforms.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(
            low_res_masks, self._orig_hw[img_idx]
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def postprocess_slices(self, slice_mask_dict, return_logits):
        """
        Postprocessing steps:
            - TODO
        """
        # Combine segmented slices into a volume with 0s for non-segmented slices
        dtype = np.float32 if return_logits else np.uint8
        segmentation = np.zeros((self.D, self.H, self.W), dtype)

        for z, low_res_mask in slice_mask_dict.items():
            if not return_logits:
                low_res_mask = (low_res_mask > 0.5).to(torch.uint8)
            segmentation[z, :, :] = low_res_mask.cpu().numpy()

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

        mask_dict = prompt.masks if prompt.masks is not None else {}
        preprocessed_prompt_dict = self.preprocess_prompt(prompt)
        slices_to_process = [
            slice_idx for slice_idx in slices_to_infer if slice_idx not in self.image_embeddings_dict.keys()
        ]

        slices_processed = self.preprocess_img(self.img, slices_to_process)

        masks_dict = {}
        low_res_logits_dict = {}

        for slice_idx in slices_to_infer:
            if slice_idx in self.image_embeddings_dict.keys():
                features = self.image_embeddings_dict[slice_idx]
            else:
                slice = slices_processed[slice_idx]
                with torch.no_grad():
                    features = self.get_features(slice)
                features_cpu = {k:v.cpu() for k,v in features.items()}
                self.image_embeddings_dict[slice_idx] = features_cpu

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

            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                slice_points[0], slice_points[1], slice_box, slice_mask, 
            )

            # Infer
            masks_dict, iou_predictions, low_res_masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                self.multimask_output,
                return_logits=return_logits,
            )

            masks_np = masks_dict.squeeze(0).float().detach().cpu().numpy()
            iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
            low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy() 

            masks_dict[slice_idx] = masks_np
            low_res_logits_dict[slice_idx] = low_res_masks_np

        segmentation = self.postprocess_slices(masks_dict, return_logits)

        # Fill in missing slices using a previous segmentation if desired
        if prev_seg is not None:
            segmentation = self.merge_seg_with_prev_seg(segmentation, prev_seg, slices_to_infer)

        # Reorient to original orientation and return with metadata
        # Turn into Nifti object in original space
        segmentation_model_arr = segmentation
        segmentation_orig = self.inv_trans_dense(segmentation)

        return segmentation_orig, low_res_logits_dict, segmentation_model_arr

