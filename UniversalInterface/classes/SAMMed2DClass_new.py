import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2

from utils.base_classes import SegmenterWrapper, Inferer, Points, Boxes2d

class SAMMed2DWrapper(SegmenterWrapper):
    def __init__(self, model):
        self.device = model.device
        self.model = model.to(self.device)
        self.multimask_output = True # Hardcoded to match defaults from original
    
    @torch.no_grad()
    def __call__(self, points, box, mask, image_embedding):
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=box,
            masks=mask,
        )

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings = image_embedding,
            image_pe = self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
        
        if self.multimask_output:
            max_values, max_indexs = torch.max(iou_predictions, dim=1)
            max_values = max_values.unsqueeze(1)
            iou_predictions = max_values
            low_res_masks = low_res_masks[:, max_indexs]
        
        return(low_res_masks)
    
    # def __call__(
    #     self,
    #     slice,
    #     point_coords,
    #     point_labels
    # ):
    #     img_embedding = self.model.image_encoder(slice.to(self.device))
        
    #     sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
    #         points=(point_coords, point_labels),
    #         boxes=None,
    #         masks=None,
    #     )

    #     # Predict masks
    #     low_res_masks, iou_predictions = self.model.mask_decoder(
    #         image_embeddings=img_embedding,
    #         image_pe=self.model.prompt_encoder.get_dense_pe(),
    #         sparse_prompt_embeddings=sparse_embeddings,
    #         dense_prompt_embeddings=dense_embeddings,
    #         multimask_output=self.multimask_output,
    #     )
    #     self.image_embedding = img_embedding
    #     self.sparse_embeddings = sparse_embeddings
    #     self.dense_prompt_embeddings = dense_embeddings
    #     self.low_res_masks = low_res_masks
    #     self.iou_predictions = iou_predictions

    #     if self.multimask_output:
    #         max_values, max_indexs = torch.max(iou_predictions, dim=1)
    #         max_values = max_values.unsqueeze(1)
    #         iou_predictions = max_values
    #         low_res_masks = low_res_masks[:, max_indexs]

    #     # Upscale the masks to the original image resolution
    
        
    #     return low_res_masks
    
class SAMMed2DInferer(Inferer):
    supported_prompts = (Points,)

    def __init__(self, model):
        self.segmenter = SAMMed2DWrapper(model)
        self.inputs = None
        self.logit_threshold = 0 # Hardcoded
        self.device = model.device
        self.new_size = (self.segmenter.model.image_encoder.img_size, self.segmenter.model.image_encoder.img_size)
        self.image_embeddings_dict = {}
        self.verbose = True # Can change directly if desired

        self.pixel_mean, self.pixel_std = self.segmenter.model.pixel_mean.squeeze().cpu().numpy(), self.segmenter.model.pixel_std.squeeze().cpu().numpy()

    def clear_embeddings(self):
        self.image_embeddings_dict = {}
        print('Embeddings cleared')

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

    def apply_boxes(self, boxes, original_size, new_size): # Copied over from SAM-Med2D predictor_sammed.py
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
        return boxes.reshape(-1, 4)

    def preprocess_img(self, img, slices_to_infer):
        img = np.repeat(img[..., None], repeats=3, axis=-1) #  Add channel dimension and make RGB giving DHWC
        
        slices_processed = {}
        for slice_idx in slices_to_infer:
            slice = img[slice_idx, ...]
            
            slice = ((slice-slice.min())/(slice.max()-slice.min() + 1e-6)*255).astype(np.uint8) # Get slice into [0,255] rgb scale
            slice = (slice-self.pixel_mean)/self.pixel_std # normalise

            transforms = self.transforms(self.new_size)
            augments = transforms(image=slice)
            slice = augments['image'][None, :, :, :] # Add batch dimension

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
                slice_coords, slice_labs = slice_coords.unsqueeze(0).to(self.device), slice_labs.unsqueeze(0).to(self.device) # add batch dimension, move to device.
                preprocessed_prompts_dict[slice_idx] = (slice_coords, slice_labs)

        if isinstance(prompt, Boxes2d):
            slices_to_infer = prompt.get_slices_to_infer()
            preprocessed_prompts_dict = {}
            for slice_index, box in prompt.value.items():
                box = np.array(box)
                box = self.apply_boxes(box, (self.H, self.W), self.new_size)
                box = torch.as_tensor(box, dtype=torch.float, device=self.device)
                box = box[None, :]
                preprocessed_prompts_dict[slice_index] = box.to(self.device)
            
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
            mask = F.interpolate(low_res_mask, self.new_size, mode="bilinear", align_corners=False)
            mask = F.interpolate(mask, self.original_size, mode="bilinear", align_corners=False) # upscale in two steps to match original code

            segmentation[z,:,:] = mask
        segmentation = (torch.sigmoid(segmentation) > 0.5).numpy()
        segmentation = segmentation.astype(np.uint8)

        return(segmentation)
    
    def predict(self, img, prompt):
        if isinstance(prompt, Points):
            self.prompt_type = 'point'
        elif isinstance(prompt, Boxes2d):
            self.prompt_type = 'box'
        else:
            raise RuntimeError(f'Currently only points and boxes are supported, got {type(prompt)}')
        
        if self.verbose and self.image_embeddings_dict != {}:
            print('Using previously generated image embeddings')

        img, prompt = deepcopy(img), deepcopy(prompt)
        
        self.D, self.H, self.W = img.shape
        self.original_size = (self.H, self.W)
        
        
        preprocessed_prompt_dict, slices_to_infer = self.preprocess_prompt(prompt)
        slices_processed = self.preprocess_img(img, slices_to_infer)

        slice_mask_dict = {}
        if self.verbose:
            slices_to_infer = tqdm(slices_to_infer, desc = 'Performing inference on slices')
        for slice_idx in slices_to_infer:
            # Get image embedding (either create it, or read it if stored)
            if slice_idx in self.image_embeddings_dict.keys():
                image_embedding = self.image_embeddings_dict[slice_idx]
            else:
                slice = slices_processed[slice_idx]
                with torch.no_grad():
                    image_embedding = self.segmenter.model.image_encoder(slice.to(self.device))
                self.image_embeddings_dict[slice_idx] = image_embedding

            # Get prompts
            slice_points, slice_box, slice_mask = None, None, None # Initialise to empty

            if self.prompt_type == 'point':
                slice_points = preprocessed_prompt_dict[slice_idx]

            if self.prompt_type == 'box':
                slice_box = preprocessed_prompt_dict[slice_idx]

            # Infer
            slice_raw_outputs = self.segmenter(points = slice_points, box=slice_box, mask = slice_mask, image_embedding = image_embedding) # Add batch dimensions
            slice_mask_dict[slice_idx] = slice_raw_outputs
            
        segmentation = self.postprocess_slices(slice_mask_dict)

        return(segmentation)
    

