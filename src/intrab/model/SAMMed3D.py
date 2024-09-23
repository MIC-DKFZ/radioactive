from pathlib import Path
from pathlib import Path
from loguru import logger
import torch
import numpy as np
import torch.nn.functional as F
from typing import TypeVar
import torchio as tio
from itertools import product
import nibabel as nib


from intrab.model.inferer import Inferer
from intrab.prompts.prompt import PromptStep
from intrab.utils.SAMMed3D_segment_anything.build_sam3D import build_sam3D_vit_b_ori
from intrab.utils.SAMMed3D_segment_anything.modeling.sam3D import Sam3D
from intrab.utils.image import get_crop_pad_params_from_gt_or_prompt
from intrab.utils.resample import get_current_spacing_from_affine, resample_to_shape, resample_to_spacing
from intrab.utils.transforms import resample_to_spacing_sparse
from intrab.datasets_preprocessing.conversion_utils import load_any_to_nib


def load_sammed3d(checkpoint_path, device="cuda"):
    sam_model_tune = build_sam3D_vit_b_ori(checkpoint=None)
    if checkpoint_path is not None:
        model_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = model_dict["model_state_dict"]
        sam_model_tune.load_state_dict(state_dict)
        sam_model_tune.to(device)
        sam_model_tune.eval()

    return sam_model_tune


class SAMMed3DInferer(Inferer):
    dim = 3
    supported_prompts = ("point", "mask")
    offset_mode = "center"  # Changing this will require siginificant reworking of code; currently doesn't matter anyway since the other method doesn't work
    pass_prev_prompts = True
    target_spacing = (1.5, 1.5, 1.5)

    def __init__(self, checkpoint, device="cuda", use_only_first_point=True):
        super().__init__(checkpoint, device)
        self.use_only_first_point = use_only_first_point

    def segment(self, img_embedding, low_res_mask, coords, labels):
        # Get prompt embeddings
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=[coords, labels],
            boxes=None,
            masks=low_res_mask.to(self.device),
        )

        ## Decode
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=img_embedding.to(self.device),  # (B, 384, 64, 64, 64)
            image_pe=self.model.prompt_encoder.get_dense_pe(),  # (1, 384, 64, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 384)
            dense_prompt_embeddings=dense_embeddings,  # (B, 384, 64, 64, 64)
            multimask_output=False,
        )

        return low_res_logits

    def load_model(self, checkpoint_path: str | Path, device: str) -> Sam3D:
        return load_sammed3d(checkpoint_path, device)

    def set_image(self, img_path: str | Path) -> None:
        if self._image_already_loaded(img_path=img_path):
            return
        img_nib = load_any_to_nib(img_path)
        self.orig_affine = img_nib.affine
        self.orig_shape = img_nib.shape

        self.img, self.inv_trans_dense = self.transform_to_model_coords_dense(img_path, is_seg=False)
        self.new_shape = self.img.shape
        self.loaded_image = img_path

    def transform_to_model_coords_dense(self, nifti: Path | str | nib.Nifti1Image, is_seg: bool) -> np.ndarray:
        """
        Doesn't include a to_canonical call since in the training the call was broken (attempted without metadata)
        """
        if isinstance(nifti, (Path, str)):
            nifti = load_any_to_nib(nifti)
        affine = nifti.affine
        orig_shape = nifti.shape
        data = nifti.get_fdata()

        # Resample to 1.5mm, 1.5mm, 1.5mm
        orig_spacing = get_current_spacing_from_affine(affine)

        img_respaced = resample_to_spacing(data, orig_spacing, self.target_spacing, is_seg=is_seg)

        def inv_trans_dense(arr: np.ndarray):
            img_orig_spacing = resample_to_shape(
                arr, current_spacing=self.target_spacing, new_shape=orig_shape, new_spacing=orig_spacing, is_seg=True
            )
            output_nib = nib.Nifti1Image(img_orig_spacing, affine)
            return output_nib

        return img_respaced, inv_trans_dense

    def transform_to_model_coords_sparse(self, coords: np.ndarray) -> np.ndarray:
        current_spacing = get_current_spacing_from_affine(self.orig_affine)
        coords_respaced = resample_to_spacing_sparse(
            coords, current_spacing, self.target_spacing, self.new_shape, round=True
        )

        return coords_respaced

    def preprocess_img(self, img3D: np.ndarray, cropping_params, padding_params):
        img3D = torch.from_numpy(img3D)
        subject = tio.Subject(image=tio.ScalarImage(tensor=img3D.unsqueeze(0)))

        roi_shape = (128, 128, 128)
        vol_bound = (0, img3D.shape[0], 0, img3D.shape[1], 0, img3D.shape[2])
        center_oob_ori_roi = (
            cropping_params[0] - padding_params[0],
            cropping_params[0] + roi_shape[0] - padding_params[0],
            cropping_params[2] - padding_params[2],
            cropping_params[2] + roi_shape[1] - padding_params[2],
            cropping_params[4] - padding_params[4],
            cropping_params[4] + roi_shape[2] - padding_params[4],
        )
        window_list = []
        offset_dict = {
            "rounded": list(product((-32, +32, 0), repeat=3)),
            "center": [(0, 0, 0)],
        }

        for offset in offset_dict[self.offset_mode]:
            # get the position in original volume~(allow out-of-bound) for current offset
            oob_ori_roi = (
                center_oob_ori_roi[0] + offset[0],
                center_oob_ori_roi[1] + offset[0],
                center_oob_ori_roi[2] + offset[1],
                center_oob_ori_roi[3] + offset[1],
                center_oob_ori_roi[4] + offset[2],
                center_oob_ori_roi[5] + offset[2],
            )
            # get corresponing padding params based on `vol_bound`
            padding_params = [0 for i in range(6)]
            for idx, (ori_pos, bound) in enumerate(zip(oob_ori_roi, vol_bound)):
                pad_val = 0
                if idx % 2 == 0 and ori_pos < bound:  # left bound
                    pad_val = bound - ori_pos
                if idx % 2 == 1 and ori_pos > bound:
                    pad_val = ori_pos - bound
                padding_params[idx] = pad_val
            # get corresponding crop params after padding
            cropping_params = (
                oob_ori_roi[0] + padding_params[0],
                vol_bound[1] - oob_ori_roi[1] + padding_params[1],
                oob_ori_roi[2] + padding_params[2],
                vol_bound[3] - oob_ori_roi[3] + padding_params[3],
                oob_ori_roi[4] + padding_params[4],
                vol_bound[5] - oob_ori_roi[5] + padding_params[5],
            )
            # pad and crop for the original subject
            pad_and_crop = tio.Compose(
                [
                    tio.Pad(padding_params, padding_mode=0),
                    tio.Crop(cropping_params),
                ]
            )
            subject_roi = pad_and_crop(subject)
            (img3D_roi,) = subject_roi.image.data.clone().detach().unsqueeze(0)
            norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
            img3D_roi = norm_transform(img3D_roi)  # (N, C, W, H, D)
            img3D_roi = img3D_roi.unsqueeze(dim=0).to(self.device)
            patch_embedding = self.model.image_encoder(img3D_roi.to(self.device))  # (1, 384, 16, 16, 16)

            # collect all position information, and set correct roi for sliding-windows in
            # todo: get correct roi window of half because of the sliding
            windows_clip = [0 for i in range(6)]
            for i in range(3):
                if offset[i] < 0:
                    windows_clip[2 * i] = 0
                    windows_clip[2 * i + 1] = -(roi_shape[i] + offset[i])
                elif offset[i] > 0:
                    windows_clip[2 * i] = roi_shape[i] - offset[i]
                    windows_clip[2 * i + 1] = 0
            pos3D_roi = dict(
                padding_params=padding_params,
                cropping_params=cropping_params,
                ori_roi=(
                    cropping_params[0] + windows_clip[0],
                    cropping_params[0] + roi_shape[0] - padding_params[0] - padding_params[1] + windows_clip[1],
                    cropping_params[2] + windows_clip[2],
                    cropping_params[2] + roi_shape[1] - padding_params[2] - padding_params[3] + windows_clip[3],
                    cropping_params[4] + windows_clip[4],
                    cropping_params[4] + roi_shape[2] - padding_params[4] - padding_params[5] + windows_clip[5],
                ),
                pred_roi=(
                    padding_params[0] + windows_clip[0],
                    roi_shape[0] - padding_params[1] + windows_clip[1],
                    padding_params[2] + windows_clip[2],
                    roi_shape[1] - padding_params[3] + windows_clip[3],
                    padding_params[4] + windows_clip[4],
                    roi_shape[2] - padding_params[5] + windows_clip[5],
                ),
            )

            window_list.append((patch_embedding, pos3D_roi))
        return window_list

    def preprocess_prompt(self, pts_prompt, cropping_params, padding_params):

        coords = pts_prompt.coords
        labels = pts_prompt.labels

        # Transform prompt in line with image transforms
        point_offset = np.array(
            [
                padding_params[0] - cropping_params[0],
                padding_params[2] - cropping_params[2],
                padding_params[4] - cropping_params[4],
            ]
        )
        coords = coords + point_offset

        batch_points = torch.from_numpy(coords).unsqueeze(0).to(self.device)
        batch_labels = torch.tensor(labels).unsqueeze(0).to(self.device)
        if (
            self.use_only_first_point
        ):  # use only the first point since the model wasn't trained to receive multiple points in one go
            batch_points = batch_points[:, :1]
            batch_labels = batch_labels[:, :1]

        return batch_points, batch_labels

    def create_or_format_low_res_logits(self, prev_low_res_logits: None | np.ndarray):
        """
        SAMMed3D Expects a logit mask (many other models just take None). If a previous mask isn't supplied, a mask of 0s of the right shape is to be created.
        """
        if prev_low_res_logits is not None:
            # if previous low res logits are present, add number and channel dimensions
            prev_low_res_logits = torch.from_numpy(prev_low_res_logits)
            low_res_logits = prev_low_res_logits.unsqueeze(0).unsqueeze(0).to(self.device)
        else:  # If no low res mask supplied, create one consisting of zeros
            low_res_spatial_shape = [
                dim // 4 for dim in (128, 128, 128)
            ]  # batch and channel dimensions remain the same, spatial dimensions are quartered
            low_res_logits = torch.zeros([1, 1] + low_res_spatial_shape).to(
                self.device
            )  # [1,1] are batch and channel dimensions

        return low_res_logits

    @torch.no_grad()
    def predict(
        self,
        prompt: PromptStep,
        crop_pad_params: tuple[tuple, tuple] | None = None,
        prev_seg: np.ndarray = None,  # Argument not used - present to align with prediction for 2d models
        promptstep_in_model_coord_system=False,
    ) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:  # If iterating, use previous patching, previous embeddings
        cheat = False
        gt = None
        if not isinstance(prompt, PromptStep):
            raise TypeError(f"Prompts must be supplied as an instance of the Prompt class.")
        if prompt.has_boxes:
            raise ValueError(f"Box prompts have been supplied, but are not supported by SAMMed3D.")

        # if len(prompt.coords) > 1:
        #     logger.warning(
        #         "SAMMed3D Can break when multiple points are passed, especially if the points are more than 128 apart in any dimension"
        #     )
        # Not needed: We take care.

        if not promptstep_in_model_coord_system:
            prompt = self.transform_promptstep_to_model_coords(prompt)

        if crop_pad_params is None:
            crop_pad_params = get_crop_pad_params_from_gt_or_prompt(self.img, prompt, cheat, gt)

        cropping_params, padding_params = crop_pad_params

        patch_list = self.preprocess_img(self.img, cropping_params, padding_params)

        prev_low_res_logits = prompt.masks
        coords, labels = self.preprocess_prompt(prompt, cropping_params, padding_params)
        if (
            crop_pad_params is not None or cheat
        ):  # Check that the prompt lies within the patch - only necessary if using a previously generated patch
            if torch.any(torch.logical_or(coords < 0, coords >= 128)):
                raise RuntimeError("Prompt coordinates do not lie within stored patch!")

        segmentation = np.zeros_like(self.img, dtype=np.uint8)
        for patch_embedding, pos3D in patch_list:
            logit_mask = self.create_or_format_low_res_logits(prev_low_res_logits)

            low_res_logits = self.segment(patch_embedding, logit_mask, coords, labels)
            logits = (
                F.interpolate(low_res_logits, size=(128, 128, 128), mode="trilinear", align_corners=False)
                .detach()
                .cpu()
                .squeeze()
            )
            seg_mask = (logits > 0.5).numpy().astype(np.uint8)
            ori_roi, pred_roi = pos3D["ori_roi"], pos3D["pred_roi"]

            seg_mask_roi = seg_mask[
                ..., pred_roi[0] : pred_roi[1], pred_roi[2] : pred_roi[3], pred_roi[4] : pred_roi[5]
            ]
            segmentation[..., ori_roi[0] : ori_roi[1], ori_roi[2] : ori_roi[3], ori_roi[4] : ori_roi[5]] = (
                seg_mask_roi
            )

        # Turn into Nifti object in original space
        segmentation_model_arr = segmentation
        segmentation_orig_nib = self.inv_trans_dense(segmentation)

        return segmentation_orig_nib, low_res_logits.detach().cpu().squeeze().numpy(), segmentation_model_arr
