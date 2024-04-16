import torch
import numpy as np
import torch.nn.functional as F

from utils.base_classes import SegmenterWrapper, Inferer, Boxes2d, Points

from tqdm import tqdm

class MedSAMWrapper(SegmenterWrapper):
    accepted_prompts = [Points]
    def __init__(self, model, device):
        self.device = device
        self.model = model.to(self.device)
        
    @torch.no_grad()
    def __call__(self, img_1024_tensor, box_1024):
        with torch.no_grad():
            img_embed = self.model.image_encoder(img_1024_tensor) # (1, 256, 64, 64)

        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=img_embed, # (B, 256, 64, 64)
            image_pe=self.model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        low_res_pred = low_res_pred.squeeze().cpu()  # (256, 256)
        return low_res_pred
    
class MedSAMInferer(Inferer):
    supported_prompts = (Boxes2d,)

    def __init__(self, segmenter_wrapper: MedSAMWrapper, device = 'cuda'):
        self.segmenter = segmenter_wrapper
        self.inputs = None
        self.logit_threshold = 0.5 # Hardcoded
        self.device = device

    def preprocess_img(self, img, slices_to_infer):
        img_new = torch.from_numpy(img).unsqueeze(0).unsqueeze(0) # add batch and channel dimensions
        img_new = torch.repeat_interleave(img_new, repeats=3, dim=1) # 1 channel -> 3 channels (align to RGB)

        slices_processed = {}
        for slice_idx in slices_to_infer:
            slice = img_new[...,slice_idx]
            slice = (slice - slice.min()) / (slice.max() - slice.min() + 1e-10)
            slice = F.interpolate(slice, (1024,1024), mode = 'bicubic', align_corners=False).clamp(0,1)
            slices_processed[slice_idx] = slice.float()

        return(slices_processed)
            
    def preprocess_prompt(self, prompt):
        if isinstance(prompt, Boxes2d):
            slices_to_infer = prompt.get_slices_to_infer()

            boxes_processed_dict = {}
            for slice_idx, box in prompt.value.items():
                #box = box[[1,0,3,2]]# Transpose to get coords in row-major format
                box_1024 = box / np.array((self.W, self.H, self.W, self.H)) * 1024
                box_torch = torch.from_numpy(box_1024).float().unsqueeze(0).unsqueeze(0) # Add 'number of boxes' and batch dimensions
                boxes_processed_dict[slice_idx] = box_torch

            return(slices_to_infer, boxes_processed_dict)

    def postprocess_slices(self, slice_mask_dict):
        '''
        Postprocessing steps:
            - Combine inferred slices into one volume, interpolating back to the original volume size
            - Turn logits into binary mask
        '''
        # Combine segmented slices into a volume with 0s for non-segmented slices
        
        segmentation = torch.zeros((self.W, self.H, self.D))
        for (z,low_res_mask) in slice_mask_dict.items():

            low_res_mask = low_res_mask.unsqueeze(0).unsqueeze(0) # Include batch and channel dimensions
            low_res_mask = F.interpolate(
                low_res_mask,
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            )  # (1, 1, gt.shape)
            segmentation[:,:,z] = low_res_mask

        segmentation = (segmentation > self.logit_threshold).numpy()
        segmentation = segmentation.astype(np.uint8)

        return(segmentation)
    
    def predict(self, img, prompt):
        if not isinstance(prompt, Boxes2d):
            raise RuntimeError('Currently only 2d bboxes are supported')
        
        self.W, self.H, self.D = img.shape

        slices_to_infer, boxes_processed = self.preprocess_prompt(prompt)
        slices_processed = self.preprocess_img(img, slices_to_infer)

        
        slice_mask_dict = {}

        for slice_idx in tqdm(slices_to_infer, desc = 'Performing inference on slices'):
            slice, box = slices_processed[slice_idx], boxes_processed[slice_idx]
            
            with torch.no_grad():
                slice_logits = self.segmenter(slice.to(self.device), box.to(self.device))

            slice_mask_dict[slice_idx] = slice_logits

            
        segmentation = self.postprocess_slices(slice_mask_dict)

        return(segmentation)
