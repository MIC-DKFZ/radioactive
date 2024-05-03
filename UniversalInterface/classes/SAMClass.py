import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from typing import TypeVar
from copy import deepcopy

from utils.base_classes import Points, Inferer, SegmenterWrapper

import utils.promptUtils as prUt
import utils.imageUtils as imUt

SAM = TypeVar('SAM')


# class AbstractInteractiveModel(ABC):

#     def __call__(self, prompt: Prompt)


class SAMWrapper(SegmenterWrapper):
    def __init__(self, model: SAM, device):
        self.device = device
        self.model = model.to(self.device)

    def __call__(self, slice, coords, labs):
        with torch.no_grad():
            image_embedding = self.model.image_encoder(slice)

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=(coords, labs),
                boxes=None,
                masks=None,
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

    def __init__(self, segmenter_wrapper: SAMWrapper, device = 'cuda'):
        self.segmenter = segmenter_wrapper
        self.prev_mask = None
        self.target_volume_shape = 128 # Hardcoded to match training
        self.target_slice_shape = 256 # Hardcoded to match training
        self.inputs = None
        self.mask_threshold = 0 
        self.device = device

    def preprocess_img(self, img, slices_to_infer):
        '''
        Preprocessing steps
            - crop and pad to target_volume_shape
            - Cast to torch tensor with batch size 1 and rgb channels, ie shape [1, 3, *target_volume_shape]
            - Slice based processing:
                - standardise per slice (no masking)
                - interpolate to target_slice_shape
        '''
        img = imUt.crop_im(img, self.crop_params) 
        img = imUt.pad_im(img, self.pad_params)

        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0) # add batch and channel dimensions
        img = torch.repeat_interleave(img, repeats=3, dim=1) # 1 channel -> 3 channels (align to RGB)

        # Perform slicewise processing and collect back into a volume at the end
        slices_processed = {}
        for slice_idx in slices_to_infer:
            slice = img[:, :, slice_idx, ...]
            slice = (slice - slice.min()) / (slice.max() - slice.min() + 1e-10) * 255.
            slice = F.interpolate(slice, (self.target_slice_shape, self.target_slice_shape), mode = 'bilinear', align_corners=False)
            slices_processed[slice_idx] = slice.float()
        
        return(slices_processed)
    
    def preprocess_points(self, prompt):
        '''
        Preprocessing steps:
            - Modify in line with the volume cropping
            - Modify in line with the interpolation
            - Collect into a dictionary of slice:slice prompt
        '''
        coords = prompt.value['coords']
        labs = prompt.value['labels']
        # Bring coordinates from original coordinate system to cropped coordinate system
        coords = prUt.crop_pad_coords(coords, self.crop_params, self.pad_params)

        keep_inds = [i for i, c in enumerate(coords) if 0 <= c[0] < 128]
        if len(keep_inds) != len(coords):
            print('Warning: foreground does not fit into crop') # Perhaps raise more formally
            coords = coords[keep_inds]
            labs = [labs[i] for i in keep_inds]

        slices_to_infer = set(coords[:,0].astype(int)) # zeroth element of batch (of size one), all triples, z coordinate # Can't use the usual get_slices_to_infer since we've changed the points. Should just modify the prompt and use the method.

        # Bring coordinates from post-crop coordinate system to post interpolation coordinate system
        coords_resized = np.array([[p[0],
                                    np.round((p[1]+0.5)/self.target_volume_shape * self.target_slice_shape),
                                    np.round((p[2]+0.5)/self.target_volume_shape * self.target_slice_shape), 
                                    ] for p in coords])

        coords_resized = coords_resized[:,[0,2,1]] # Transpose x and y 

        # Convert to torch tensor 
        coords_resized = torch.from_numpy(coords_resized).int()
        labs = torch.tensor(labs)

        # Collate
        preprocessed_points_dict = {}
        for slice_idx in slices_to_infer:
            slice_coords_mask = (coords_resized[:,0] == slice_idx)
            slice_coords, slice_labs = coords_resized[slice_coords_mask, 1:], labs[slice_coords_mask] # Leave out z coordinate in slice_coords
            preprocessed_points_dict[slice_idx] = (slice_coords, slice_labs)

        return(preprocessed_points_dict, slices_to_infer)
    
    def postprocess_slices(self, slice_mask_dict):
        '''
        Postprocessing steps:
            - Combine inferred slices into one volume, interpolating back to the cropped volume size (ie 128,128,128)
            - Turn logits into binary mask
            - Invert crop/pad to get back to original image dimensions
        '''
        # Combine segmented slices into a volume with 0s for non-segmented slices
        segmentation = torch.zeros((self.target_volume_shape, self.target_volume_shape, self.target_volume_shape))

        for (z,low_res_mask) in slice_mask_dict.items():
            low_res_mask = low_res_mask.unsqueeze(0).unsqueeze(0) # Include batch and channel dimensions
            slice_mask = F.interpolate(low_res_mask, (self.target_volume_shape, self.target_volume_shape), mode="bilinear", align_corners=False)
            segmentation[z,:,:] = slice_mask

        segmentation = (segmentation > self.mask_threshold).numpy()
        segmentation = segmentation.astype(np.uint8)

        # Bring patch 
        segmentation = imUt.invert_crop_or_pad(segmentation, self.crop_params, self.pad_params)

        return(segmentation)
 
    def predict(self, img, prompt):
        if not isinstance(prompt, Points):
            raise RuntimeError('Currently only points are supported')

        prompt = deepcopy(prompt)
        #prompt.value['coords']=prompt.value['coords'][:,::-1]

        self.crop_pad_center = prUt.get_crop_pad_center_from_points(prompt)
        
        #self.crop_pad_center = self.crop_pad_center[::-1]
        
        self.crop_params, self.pad_params = imUt.get_crop_pad_params(img, self.crop_pad_center, (self.target_volume_shape, self.target_volume_shape, self.target_volume_shape))
        
        
        preprocessed_points_dict, slices_to_infer = self.preprocess_points(prompt)
        self.preprocessed_points_dict = preprocessed_points_dict
        slices_processed = self.preprocess_img(img, slices_to_infer)
        self.slices_processed = slices_processed
        
        slice_mask_dict = {}
        for slice_idx in tqdm(slices_to_infer, desc = 'Performing inference on slices'):
            slice = slices_processed[slice_idx]
            
            slice_coords, slice_labs = preprocessed_points_dict[slice_idx]

            # slice = slice.permute(0,1,3,2)
            #slice_coords = slice_coords[:,[1,0]]

            # Infer
            slice_raw_outputs = self.segmenter(slice = slice.to(self.device), coords = slice_coords.unsqueeze(0).to(self.device), labs = slice_labs.unsqueeze(0).to(self.device)) # Add batch dimensions
            slice_mask_dict[slice_idx] = slice_raw_outputs
            
        segmentation = self.postprocess_slices(slice_mask_dict)

        return(segmentation)
