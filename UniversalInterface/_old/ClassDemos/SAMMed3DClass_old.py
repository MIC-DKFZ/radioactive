import torch
import numpy as np
import torch.nn.functional as F
from typing import TypeVar
from copy import deepcopy

from utils.base_classes import Points, Inferer, SegmenterWrapper

import utils.image as imUt
import utils.prompt as prUt

SAM3D = TypeVar('SAM3D')


class SAMMed3DWrapper(SegmenterWrapper):
    def __init__(self, model: SAM3D, device):
        self.model = model.to(device)
        self.device = device

    def __call__(self, img, prompt):
        # Get prompt embeddings
        ## Initialise empty prompts 
        coords, labs = None, None
        boxes = None

        ## Fill with relevant prompts
        if isinstance(prompt, Points):
            coords, labs = prompt.value['coords'], prompt.value['labels']

        low_res_spatial_shape = [dim//4 for dim in img.shape[-3:]] #batch and channel dimensions remain the same, spatial dimensions are quartered 
        low_res_mask = torch.zeros([1,1] + low_res_spatial_shape).to(self.device) # [1,1] is batch and channel dimensions

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points = [coords, labs],
                boxes = boxes,
                masks = low_res_mask.to(self.device),
            )

        # Get image mebedding
        
        with torch.no_grad():
            image_embedding = self.model.image_encoder(img) # (1, 384, 16, 16, 16)        
        
        ## Decode
        mask_out, _ = self.model.mask_decoder(
            image_embeddings = image_embedding.to(self.device), # (B, 384, 64, 64, 64)
            image_pe = self.model.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
            sparse_prompt_embeddings = sparse_embeddings, # (B, 2, 384)
            dense_prompt_embeddings = dense_embeddings, # (B, 384, 64, 64, 64)
            multimask_output = False,
            )
        

        logits = F.interpolate(mask_out, size=img.shape[-3:], mode = 'trilinear', align_corners = False).detach().cpu().squeeze()
        
        return(logits)

class SAMMed3DInferer(Inferer):
    supported_prompts = (Points,)

    required_shape = (128, 128, 128) # Hard code to match training
    logit_threshold = 0.5

    def __init__(self, segmenter_wrapper: SAMMed3DWrapper, device = 'cuda'):
        self.segmenter = segmenter_wrapper
        self.device = device

    def preprocess_img(self, img, crop_params, pad_params):
        '''Any necessary preprocessing steps'''

        img = imUt.crop_im(img, crop_params) 
        img = imUt.pad_im(img, pad_params)

        mask = img > 0
        mean, std = img[mask].mean(), img[mask].std()
        # standardize_func = tio.ZNormalization(masking_method=lambda x: x > 0)
        # img2 = np.array(standardize_func(torch.from_numpy(img).unsqueeze(0))).squeeze(0) # Gives a different result; investigate
        img = (img-mean)/std 

        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device) # add channel and batch dimensions
        return(img)
    
    def preprocess_prompt(self, prompt, crop_params: list, pad_params: list): 
        if isinstance(prompt, Points):
            prompt.value['coords'] = prUt.crop_pad_coords(prompt.value['coords'], crop_params, pad_params)

            prompt.value['coords'] = torch.from_numpy(prompt.value['coords']).unsqueeze(0).to(self.device) # Must have shape B, N, 3
            prompt.value['labels'] = torch.tensor(prompt.value['labels']).unsqueeze(0).to(self.device)
            return(prompt)
    
    def postprocess_logits(self, mask: np.array, cropping_params, padding_params):
        mask = torch.sigmoid(mask)
        mask = (mask>self.logit_threshold).numpy().astype(np.uint8)
        mask = imUt.invert_crop_or_pad(mask, cropping_params, padding_params)

        return(mask)
 
    def predict(self, img, prompt):
        if not isinstance(prompt, SAMMed3DInferer.supported_prompts):
            raise ValueError(f'Unsupported prompt type: got {type(prompt)}')
        img, prompt = deepcopy(img), deepcopy(prompt)

        self.crop_pad_center = prUt.get_crop_pad_center_from_points(prompt)
        self.crop_params, self.pad_params = imUt.get_crop_pad_params(img, self.crop_pad_center, self.required_shape)
        prompt = self.preprocess_prompt(prompt, self.crop_params, self.pad_params)
        img = self.preprocess_img(img, self.crop_params, self.pad_params)
        
        logits = self.segmenter(img, prompt)

        segmentation = self.postprocess_logits(logits, self.crop_params, self.pad_params)

        return(segmentation)
    