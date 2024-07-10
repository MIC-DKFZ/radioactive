import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm

from utils.base_classes import SegmenterWrapper, Inferer, Boxes, Points
from utils.MedSAM_segment_anything import sam_model_registry as registry_medsam


def load_medsam(checkpoint_path, device = 'cuda'):
    medsam_model = registry_medsam['vit_b'](checkpoint=checkpoint_path)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    return(medsam_model)

class MedSAMWrapper(SegmenterWrapper):
    def __init__(self, model):
        self.model = model
        
    @torch.no_grad()
    def __call__(self, points, box, mask, image_embedding):
        # if len(box_torch.shape) == 2:
        #     box_torch = box_torch[:, None, :] # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=box,
            masks=mask,
        )

        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        return low_res_pred
    
class MedSAMInferer(Inferer):

    def __init__(self, checkpoint_path, device):
        model = load_medsam(checkpoint_path, device)
        self.segmenter = MedSAMWrapper(model)
        self.logit_threshold = 0.5 
        self.device = device
        self.image_embeddings_dict = {}
        self.verbose = True

    def preprocess_img(self, img, slices_to_process):
        slices_processed = {}
        for slice_idx in slices_to_process:
            slice = img[slice_idx,...]
            slice = (slice - slice.min()) / (slice.max() - slice.min() + 1e-10)
            slice = np.repeat(slice[None,...], repeats=3, axis=0) # dimensions are NCd_1d_2 to conform to F.interpolate

            slice = torch.from_numpy(slice).unsqueeze(0)
            slice = F.interpolate(slice, (1024,1024), mode = 'bicubic', align_corners=False).clamp(0,1)
            slices_processed[slice_idx] = slice.float()

        return(slices_processed)
            
    def preprocess_prompt(self, prompt):
        if isinstance(prompt, Boxes):
            slices_to_infer = prompt.get_slices_to_infer()

            preprocessed_prompt_dict = {}
            for slice_idx, box in prompt.value.items():
                box_1024 = box / np.array((self.W, self.H, self.W, self.H)) * 1024
                box_torch = torch.from_numpy(box_1024).float().unsqueeze(0).unsqueeze(0) # Add 'number of boxes' and batch dimensions
                preprocessed_prompt_dict[slice_idx] = box_torch.to(self.device)

            return(preprocessed_prompt_dict, slices_to_infer)
    
    def postprocess_slices(self, slice_mask_dict):
        '''
        Postprocessing steps:
            - Combine inferred slices into one volume, interpolating back to the original volume size
            - Turn logits into binary mask
        '''
        # Combine segmented slices into a volume with 0s for non-segmented slices
        segmentation = torch.zeros((self.D, self.H, self.W))
        for (z,low_res_mask) in slice_mask_dict.items():

            low_res_mask = F.interpolate(
                low_res_mask,
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            )  # (1, 1, gt.shape)
            segmentation[z,:,:] = low_res_mask

        segmentation = (segmentation > self.logit_threshold).numpy()
        segmentation = segmentation.astype(np.uint8)

        return(segmentation)
    
    def predict(self, img, prompt):
        if not isinstance(prompt, Boxes):
            raise RuntimeError(f'Currently only points and boxes are supported, got {type(prompt)}')            
        
        if self.verbose and self.image_embeddings_dict != {}:
            print('Using previously generated image embeddings')

        self.D, self.H, self.W = img.shape

        preprocessed_prompt_dict, slices_to_infer = self.preprocess_prompt(prompt)
        slices_to_process = [slice_idx for slice_idx in slices_to_infer if slice_idx not in self.image_embeddings_dict.keys()]
        slices_processed = self.preprocess_img(img, slices_to_process)

        # debugging
        self.slices_processed = slices_processed
        self.preprocessed_prompt_dict = preprocessed_prompt_dict
        
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

            # Get prompts
            slice_points, slice_box, slice_mask = None, None, None # Initialise to empty

            if isinstance(prompt, Boxes):
                slice_box = preprocessed_prompt_dict[slice_idx]

            # Infer
            slice_raw_outputs = self.segmenter(points = slice_points, box=slice_box, mask = slice_mask, image_embedding = image_embedding) # Add batch dimensions
            slice_mask_dict[slice_idx] = slice_raw_outputs

            
        segmentation = self.postprocess_slices(slice_mask_dict)

        return(segmentation)
