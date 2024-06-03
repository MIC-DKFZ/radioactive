import torch
import numpy as np
import torch.nn.functional as F
from typing import TypeVar
from copy import deepcopy
import torchio as tio
from itertools import product

from utils.base_classes import Points, Inferer, SegmenterWrapper

SAM3D = TypeVar('SAM3D')


class SAMMed3DWrapper(SegmenterWrapper):
    def __init__(self, model, device):
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

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points = [coords, labs],
                boxes = boxes,
                masks = low_res_mask.to(self.device),
            )

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
    supported_prompts = supported_prompts = (Points,)
    required_shape = (128, 128, 128) # Hard code to match training

    def __init__(self, segmenter_wrapper: SAMMed3DWrapper, use_only_first_point = False):
        self.segmenter = segmenter_wrapper
        self.device = segmenter_wrapper.device
        self.use_only_first_point = use_only_first_point
        self.offset_mode = 'center'

    def preprocess_into_patches(self, img3D, prompt = None, cheat = False, gt = None):
        img3D = torch.from_numpy(img3D)

        subject = tio.Subject(
            image = tio.ScalarImage(tensor=img3D.unsqueeze(0))
        )
        
        if cheat:
            subject.add_image(tio.LabelMap(tensor = gt.unsqueeze(0),
                                        affine = subject.image.affine,),
                            image_name = 'label')
            crop_transform = tio.CropOrPad(mask_name='label', 
                                target_shape=(128,128,128))
        else:
            coords = prompt.value['coords']
            crop_mask = torch.zeros_like(subject.image.data)
            crop_mask[0, *coords.T] = 1 # Include initial 0 for the additional N axis
            subject.add_image(tio.LabelMap(tensor = crop_mask,
                                            affine = subject.image.affine),
                                image_name="crop_mask")
            crop_transform = tio.CropOrPad(mask_name='crop_mask', 
                                    target_shape=(128,128,128))
            

        padding_params, cropping_params = crop_transform.compute_crop_or_pad(subject)
        # cropping_params: (x_start, x_max-(x_start+roi_size), y_start, ...)
        # padding_params: (x_left_pad, x_right_pad, y_left_pad, ...)
        if(cropping_params is None): cropping_params = (0,0,0,0,0,0)
        if(padding_params is None): padding_params = (0,0,0,0,0,0)
        roi_shape = crop_transform.target_shape
        vol_bound = (0, img3D.shape[0], 0, img3D.shape[1], 0, img3D.shape[2])
        center_oob_ori_roi=(
            cropping_params[0]-padding_params[0], cropping_params[0]+roi_shape[0]-padding_params[0],
            cropping_params[2]-padding_params[2], cropping_params[2]+roi_shape[1]-padding_params[2],
            cropping_params[4]-padding_params[4], cropping_params[4]+roi_shape[2]-padding_params[4],
        )
        window_list = []
        offset_dict = {
            "rounded": list(product((-32,+32,0), repeat=3)),
            "center": [(0,0,0)],
        }
        for offset in offset_dict[self.offset_mode]:
            # get the position in original volume~(allow out-of-bound) for current offset
            oob_ori_roi = (
                center_oob_ori_roi[0]+offset[0], center_oob_ori_roi[1]+offset[0],
                center_oob_ori_roi[2]+offset[1], center_oob_ori_roi[3]+offset[1],
                center_oob_ori_roi[4]+offset[2], center_oob_ori_roi[5]+offset[2],
            )
            # get corresponing padding params based on `vol_bound`
            padding_params = [0 for i in range(6)]
            for idx, (ori_pos, bound) in enumerate(zip(oob_ori_roi, vol_bound)):
                pad_val = 0
                if(idx%2==0 and ori_pos<bound): # left bound
                    pad_val = bound-ori_pos
                if(idx%2==1 and ori_pos>bound):
                    pad_val = ori_pos-bound
                padding_params[idx] = pad_val
            # get corresponding crop params after padding
            cropping_params = (
                oob_ori_roi[0]+padding_params[0], vol_bound[1]-oob_ori_roi[1]+padding_params[1],
                oob_ori_roi[2]+padding_params[2], vol_bound[3]-oob_ori_roi[3]+padding_params[3],
                oob_ori_roi[4]+padding_params[4], vol_bound[5]-oob_ori_roi[5]+padding_params[5],
            )
            # pad and crop for the original subject
            pad_and_crop = tio.Compose([
                tio.Pad(padding_params, padding_mode=crop_transform.padding_mode),
                tio.Crop(cropping_params),
            ])
            subject_roi = pad_and_crop(subject)  
            img3D_roi, = subject_roi.image.data.clone().detach().unsqueeze(0)
            norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
            img3D_roi = norm_transform(img3D_roi) # (N, C, W, H, D)
            img3D_roi = img3D_roi.unsqueeze(dim=0)
            

            # collect all position information, and set correct roi for sliding-windows in 
            # todo: get correct roi window of half because of the sliding 
            windows_clip = [0 for i in range(6)]
            for i in range(3):
                if(offset[i]<0):
                    windows_clip[2*i] = 0
                    windows_clip[2*i+1] = -(roi_shape[i]+offset[i])
                elif(offset[i]>0):
                    windows_clip[2*i] = roi_shape[i]-offset[i]
                    windows_clip[2*i+1] = 0
            pos3D_roi = dict(
                padding_params=padding_params, cropping_params=cropping_params, 
                ori_roi=(
                    cropping_params[0]+windows_clip[0], cropping_params[0]+roi_shape[0]-padding_params[0]-padding_params[1]+windows_clip[1],
                    cropping_params[2]+windows_clip[2], cropping_params[2]+roi_shape[1]-padding_params[2]-padding_params[3]+windows_clip[3],
                    cropping_params[4]+windows_clip[4], cropping_params[4]+roi_shape[2]-padding_params[4]-padding_params[5]+windows_clip[5],
                ),
                pred_roi=(
                    padding_params[0]+windows_clip[0], roi_shape[0]-padding_params[1]+windows_clip[1],
                    padding_params[2]+windows_clip[2], roi_shape[1]-padding_params[3]+windows_clip[3],
                    padding_params[4]+windows_clip[4], roi_shape[2]-padding_params[5]+windows_clip[5],
                ))

            window_list.append((img3D_roi, pos3D_roi))
        return cropping_params, padding_params, window_list

    def preprocess_prompt(self, pts_prompt):
        coords = pts_prompt.value['coords']
        labels = pts_prompt.value['labels']

        point_offset = np.array([self.padding_params[0]-self.cropping_params[0], self.padding_params[2]-self.cropping_params[2], self.padding_params[4]-self.cropping_params[4]])
        coords = coords + point_offset
        
        batch_points = torch.from_numpy(coords).unsqueeze(0).to(self.device)
        batch_labels = torch.tensor(labels).unsqueeze(0).to(self.device)
        if self.use_only_first_point: # use only the first point since the model wasn't trained to receive multiple points in one go 
            batch_points = batch_points[:, :1]
            batch_labels = batch_labels[:, :1]
        
        pts_prompt = Points(value = {'coords': batch_points, 'labels': batch_labels})
        return pts_prompt
    
    def predict(self, img, prompt, cheat = False, gt = None):
        if not isinstance(prompt, SAMMed3DInferer.supported_prompts):
            raise ValueError(f'Unsupported prompt type: got {type(prompt)}')
    
        img, prompt = deepcopy(img), deepcopy(prompt)

        self.cropping_params, self.padding_params, patch_list = self.preprocess_into_patches(img, prompt, cheat, gt)

        prompt = self.preprocess_prompt(prompt)

        pred  = np.zeros_like(img, dtype=np.uint8)
        for (image3D_patch, pos3D) in patch_list:
            image3D_patch = image3D_patch.to(self.device)
            logits = self.segmenter(image3D_patch, prompt)
            seg_mask = (logits>0.5).numpy().astype(np.uint8)
            ori_roi, pred_roi = pos3D["ori_roi"], pos3D["pred_roi"]
            
            seg_mask_roi = seg_mask[..., pred_roi[0]:pred_roi[1], pred_roi[2]:pred_roi[3], pred_roi[4]:pred_roi[5]]
            pred[..., ori_roi[0]:ori_roi[1], ori_roi[2]:ori_roi[3], ori_roi[4]:ori_roi[5]] = seg_mask_roi
        
        return(pred)