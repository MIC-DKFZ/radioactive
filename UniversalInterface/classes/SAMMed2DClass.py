import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2

from utils.base_classes import SegmenterWrapper, Inferer, Points



class SAMMed2DWrapper(SegmenterWrapper):
    def __init__(self, model, device):
        self.device = device
        self.model = model.to(self.device)
        self.multimask_output = True # Hardcoded to match defaults from original
    
    @torch.no_grad()
    def __call__(
        self,
        slice,
        point_coords,
        point_labels
    ):
        img_embedding = self.model.image_encoder(slice.to(self.device))
        
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=img_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
        )
        # self.image_embedding = img_embedding
        # self.sparse_embeddings = sparse_embeddings
        # self.dense_prompt_embeddings = dense_embeddings
        # self.low_res_masks = low_res_masks
        # self.iou_predictions = iou_predictions

        if self.multimask_output:
            max_values, max_indexs = torch.max(iou_predictions, dim=1)
            max_values = max_values.unsqueeze(1)
            iou_predictions = max_values
            low_res_masks = low_res_masks[:, max_indexs]

        # Upscale the masks to the original image resolution
    
        
        return low_res_masks
    
class SAMMed2DInferer(Inferer):
    supported_prompts = (Points,)

    def __init__(self, segmenter_wrapper: SAMMed2DWrapper, device = 'cuda'):
        self.segmenter = segmenter_wrapper
        self.inputs = None
        self.logit_threshold = 0 # Hardcoded
        self.device = device
        self.new_size = (self.segmenter.model.image_encoder.img_size, self.segmenter.model.image_encoder.img_size)

        if self.segmenter.model.pixel_mean.device.type == 'cuda':
            self.pixel_mean, self.pixel_std = self.segmenter.model.pixel_mean.squeeze().cpu().numpy(), self.segmenter.model.pixel_std.squeeze().cpu().numpy()
        else:
            self.pixel_mean, self.pixel_std = self.segmenter.model.pixel_mean.numpy(), self.segmenter.model.pixel_std.numpy()

        #self.pixel_mean, self.pixel_std = torch.from_numpy(self.pixel_mean), torch.from_numpy(self.pixel_std)

    def transforms(self, new_size): # Copied over from SAM-Med2D predictor_sammed.py
        Transforms = []
        new_h, new_w = new_size
        Transforms.append(A.Resize(int(new_h), int(new_w), interpolation=cv2.INTER_NEAREST)) # note nearest neighbour interpolation.
        Transforms.append(ToTensorV2(p=1.0))
        return A.Compose(Transforms, p=1.)
    
    def apply_coords(self, coords, original_size, new_size): # Copied over from SAM-Med2D predictor_sammed.py
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)

        return coords

    def preprocess_img(self, img, slices_to_infer):
        img = np.repeat(img[..., None], repeats=3, axis=-1) #  Add channel dimension and make RGB giving DHWC
        
        slices_processed = {}
        for slice_idx in slices_to_infer:
            slice = img[slice_idx, ...]
            
            slice = ((slice-slice.min())/(slice.max()-slice.min() + 1e-6)*255).astype(np.uint8) # Get slice into [0,255] rgb scale
            slice = (slice-self.pixel_mean)/self.pixel_std # normalise

            transforms = self.transforms(self.new_size)
            augments = transforms(image=slice)
            slice = augments['image'][None, :, :, :]

            # slice = F.interpolate(slice, (256,256), mode = 'nearest') # Note the nearest neighbours interpolation! Follows from official repo SAM-MEd2D/segment_anything/predictor_sammed.py transforms
            slices_processed[slice_idx] = slice.float()

        return(slices_processed)
            
    def preprocess_prompt(self, prompt):
        '''
        Preprocessing steps:
            - Modify in line with the volume cropping
            - Modify in line with the interpolation
            - Collect into a dictionary of slice:slice prompt
        '''
        if isinstance(prompt, Points):
            coords, labs = prompt.value['coords'], prompt.value['labels']
            coords, labs = np.array(coords).astype(float), np.array(labs).astype(int)

            slices_to_infer = set(coords[:,0].astype(int)) # zeroth element of batch (of size one), all triples, z coordinate 

            coords = coords[:,[2,1,0]] # Change from ZYX to XYZ
            coords_resized = self.apply_coords(coords, (self.H, self.W), self.new_size)

            # Convert to torch tensor 
            coords_resized = torch.as_tensor(coords_resized, dtype=torch.float)
            labs = torch.as_tensor(labs, dtype = int)

            # Collate
            preprocessed_prompts_dict = {}
            for slice_idx in slices_to_infer:
                slice_coords_mask = (coords_resized[:,2] == slice_idx)
                slice_coords, slice_labs = coords_resized[slice_coords_mask, :2], labs[slice_coords_mask] # Leave out z coordinate in slice_coords
                preprocessed_prompts_dict[slice_idx] = (slice_coords, slice_labs)

            return(preprocessed_prompts_dict, slices_to_infer)
        
    def postprocess_slices(self, slice_mask_dict):
        '''
        Postprocessing steps:
            - Combine inferred slices into one volume, interpolating back to the original volume size
            - Turn logits into binary mask
            - Invert crop/pad to get back to original image dimensions
        '''
        # Combine segmented slices into a volume with 0s for non-segmented slices
    

        segmentation = torch.zeros((self.D, self.H, self.W))
        for (z,low_res_mask) in slice_mask_dict.items():
            masks = F.interpolate(low_res_mask, self.new_size, mode="bilinear", align_corners=False)
            masks = F.interpolate(masks, self.original_size, mode="bilinear", align_corners=False) # upscale in two steps to match original code

            low_res_mask = F.interpolate(
                low_res_mask,
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            )  # (1, 1, gt.shape)
            segmentation[z,:,:] = low_res_mask
        segmentation = (torch.sigmoid(segmentation) > 0.5).numpy()
        segmentation = segmentation.astype(np.uint8)

        return(segmentation)
    
    def predict(self, img, prompt):
        if not isinstance(prompt, Points):
            raise RuntimeError('Currently only points are supported')
        img, prompt = deepcopy(img), deepcopy(prompt)
        
        self.D, self.H, self.W = img.shape
        self.original_size = (self.H, self.W)
        
        
        preprocessed_prompts_dict, slices_to_infer = self.preprocess_prompt(prompt)
        
        slices_processed = self.preprocess_img(img, slices_to_infer)
        # self.slices_processed, self.preprocessed_prompts_dict = slices_processed, preprocessed_prompts_dict
        slice_mask_dict = {}
        for slice_idx in tqdm(slices_to_infer, desc = 'Performing inference on slices'):
            slice = slices_processed[slice_idx]
            
            slice_coords, slice_labs = preprocessed_prompts_dict[slice_idx]

            # Infer
            slice_raw_outputs = self.segmenter(slice.to(self.device), slice_coords.unsqueeze(0).to(self.device), slice_labs.unsqueeze(0).to(self.device)) # Add batch dimensions
            # self.low_res_masks = slice_raw_outputs

            slice_mask_dict[slice_idx] = slice_raw_outputs
            # self.slice_mask_dict = slice_mask_dict
            
        segmentation = self.postprocess_slices(slice_mask_dict)

        return(segmentation)
    

