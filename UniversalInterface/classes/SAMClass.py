import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from typing import TypeVar
from copy import deepcopy
from argparse import Namespace

from utils.SAMMed3D_segment_anything.build_sam import sam_model_registry as registry_sam
from utils.base_classes import Points, Boxes2d, Inferer, SegmenterWrapper

import utils.prompt as prUt
import utils.image as imUt
from utils.transforms import ResizeLongestSide

SAM = TypeVar('SAM')

def load_sam(checkpoint_path, device = 'cuda', image_size = 1024):
    args = Namespace()
    args.image_size = image_size
    args.sam_checkpoint = checkpoint_path
    args.model_type = 'vit_h'
    model = registry_sam[args.model_type](args).to(device)
    return(model)

class SAMWrapper(SegmenterWrapper):
    def __init__(self, model: SAM):
        self.model = model
        self.device = model.device


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
        
        # Obtain the best mask (measured by predicted iou) from the 3 returned masks
        iou_predictions = iou_predictions[0] # Indexing within batch : we have batch size 1
        max_value, max_index = torch.max(iou_predictions, dim = 0)
        best_mask = low_res_masks[0, max_index]
        
        return(best_mask)
    
    
class SAMInferer(Inferer):
    supported_prompts = (Points,) # TODO: Implement boxes

    def __init__(self, checkpoint_path, device):
        model = load_sam(checkpoint_path, device)
        self.segmenter = SAMWrapper(model)
        self.prev_mask = None
        self.target_volume_shape = 128 # Hardcoded to match training
        self.target_slice_shape = 256 # Hardcoded to match training
        self.inputs = None
        self.mask_threshold = 0 
        self.device = model.device
        self.image_embeddings_dict = {}
        self.verbose = True

        self.pixel_mean = model.pixel_mean
        self.pixel_std = model.pixel_std
        self.transform = ResizeLongestSide(model.image_encoder.img_size)
        self.input_size = None

    def clear_embeddings(self):
        self.image_embeddings_dict = {}

    def preprocess_img(self, img, slices_to_process):
        '''
        Preprocessing steps
            - Extract slice, resize (maintaining aspect ratio), pad edges
        '''

        # Perform slicewise processing and collect back into a volume at the end
        slices_processed = {}
        for slice_idx in slices_to_process:
            slice = img[slice_idx, ...] # Now HWC
            slice = ((slice - slice.min()) / (slice.max() - slice.min() + 1e-10) * 255.).astype(np.uint8) # Change to 0-255 scale
            slice = np.repeat(slice[..., None], repeats=3, axis=-1)  # Add channel dimension to make it RGB-like
            slice = self.transform.apply_image(slice)            
            slice = torch.as_tensor(slice, device = self.device)
            slice = slice.permute(2, 0, 1).contiguous()[None, :, :, :] # Change to BCHW, make memory storage contiguous.

            if self.input_size is None:
                self.input_size = tuple(slice.shape[-2:]) # Store the input size pre-padding if it hasn't been done yet
            
            
            slice = slice = (slice - self.pixel_mean) / self.pixel_std

            h, w = slice.shape[-2:]
            padh = self.segmenter.model.image_encoder.img_size - h
            padw = self.segmenter.model.image_encoder.img_size - w
            slice = F.pad(slice, (0, padw, 0, padh))

            slices_processed[slice_idx] = slice
        self.slices_processed = slices_processed
        return(slices_processed)
    
    def preprocess_prompt(self, prompt):
        '''
        Preprocessing steps:
            - Modify in line with the volume cropping
            - Modify in line with the interpolation
            - Collect into a dictionary of slice:slice prompt
        '''
        if isinstance(prompt, Points):
            coords = prompt.coords
            labs = prompt.labels

            slices_to_infer = set(coords[:,0].astype(int)) # zeroth element of batch (of size one), all triples, z coordinate 

            coords = coords[:,[2,1,0]] # Change from ZYX to XYZ
            coords_resized = self.transform.apply_coords(coords, (self.H, self.W))

            # Convert to torch tensor 
            coords_resized = torch.as_tensor(coords_resized, dtype=torch.float)
            labs = torch.as_tensor(labs, dtype = int)

            # Collate
            preprocessed_prompts_dict = {}
            for slice_idx in slices_to_infer:
                slice_coords_mask = (coords_resized[:,2] == slice_idx)
                slice_coords, slice_labs = coords_resized[slice_coords_mask, :2], labs[slice_coords_mask] # Leave out z coordinate in slice_coords
                slice_coords, slice_labs = slice_coords.unsqueeze(0), slice_labs.unsqueeze(0)
                preprocessed_prompts_dict[slice_idx] = (slice_coords.to(self.device), slice_labs.to(self.device))

            return(preprocessed_prompts_dict, slices_to_infer)
        
        if isinstance(prompt, Boxes2d):
            slices_to_infer = prompt.get_slices_to_infer()
            preprocessed_prompts_dict = {}
            for slice_index, box in prompt.value.items():
                box = np.array(box)
                box = self.transform.apply_boxes(box, (self.H, self.W))
                box = torch.as_tensor(box, dtype=torch.float, device=self.device)
                box = box[None, :]
                preprocessed_prompts_dict[slice_index] = box.to(self.device)
            
            return(preprocessed_prompts_dict, slices_to_infer)

    
    def postprocess_slices(self, slice_mask_dict):
        '''
        Postprocessing steps:
            - TODO
        '''
        # Combine segmented slices into a volume with 0s for non-segmented slices
        segmentation = torch.zeros((self.D, self.H, self.W))

        for (z,low_res_mask) in slice_mask_dict.items():
            low_res_mask = low_res_mask.unsqueeze(0).unsqueeze(0) # Include batch and channel dimensions
            mask_input_res = F.interpolate(
                low_res_mask,
                (self.segmenter.model.image_encoder.img_size, self.segmenter.model.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )   # upscale low res mask to mask as in input_size
            mask_input_res = mask_input_res[..., : self.input_size[0], : self.input_size[1]] # Crop out any segmentations created in the padded sections
            slice_mask = F.interpolate(mask_input_res, self.original_size, mode="bilinear", align_corners=False)
        
            segmentation[z,:,:] = slice_mask

        segmentation = (segmentation > self.mask_threshold).numpy()
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

        prompt = deepcopy(prompt)
        
        self.D, self.H, self.W = img.shape
        self.original_size = (self.H, self.W) # Used for the transform class, which is taken from the original SAM code, hence the 2D size
        
        preprocessed_prompt_dict, slices_to_infer = self.preprocess_prompt(prompt)
        slices_to_process = [slice_idx for slice_idx in slices_to_infer if slice_idx not in self.image_embeddings_dict.keys()]
        slices_processed = self.preprocess_img(img, slices_to_process)
        
        slice_mask_dict = {}
        if self.verbose:
            slices_to_infer = tqdm(slices_to_infer, desc = 'Performing inference on slices')
        for slice_idx in slices_to_infer:
            if slice_idx in self.image_embeddings_dict.keys():
                image_embedding = self.image_embeddings_dict[slice_idx]
            else:
                slice = slices_processed[slice_idx]
                with torch.no_grad():
                    image_embedding = self.segmenter.model.image_encoder(slice.to(self.device))
                self.image_embeddings_dict[slice_idx] = image_embedding
            

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
