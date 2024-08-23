import torch
import numpy as np
import torch.nn.functional as F
from typing import TypeVar
from copy import deepcopy
import torchio as tio
from itertools import product
import nibabel as nib
from nibabel.orientations import io_orientation, ornt_transform

from utils.base_classes import Points, Inferer
from utils.SAMMed3D_segment_anything.build_sam3D import build_sam3D_vit_b_ori

SAM3D = TypeVar('SAM3D')

def load_sammed3d(checkpoint_path, device = 'cuda'):
    sam_model_tune = build_sam3D_vit_b_ori(checkpoint=None)
    if checkpoint_path is not None:
        model_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = model_dict['model_state_dict']
        sam_model_tune.load_state_dict(state_dict)
        sam_model_tune.to(device)
        sam_model_tune.eval()

    return (sam_model_tune)

class SAMMed3DInferer(Inferer):
    dim = 3
    supported_prompts = ('point',)
    required_shape = (128, 128, 128) # Hard code to match training
    offset_mode = 'center' # Changing this will require siginificant reworking of code; currently doesn't matter anyway since the other method doesn't work
    pass_prev_prompts = True

    def __init__(self, checkpoint_path, device, use_only_first_point = False):
        self.model = load_sammed3d(checkpoint_path,  device)
        self.device = device
        self.use_only_first_point = use_only_first_point
        self.image_set = False
        self.stored_cropping_params, self.stored_padding_params, self.stored_patch_list = None, None, None

    def segment(self, img_embedding, low_res_mask, coords, labels):
        # Get prompt embeddings
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points = [coords, labels],
            boxes = None,
            masks = low_res_mask.to(self.device),
        )        
    
        ## Decode
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings = img_embedding.to(self.device), # (B, 384, 64, 64, 64)
            image_pe = self.model.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
            sparse_prompt_embeddings = sparse_embeddings, # (B, 2, 384)
            dense_prompt_embeddings = dense_embeddings, # (B, 384, 64, 64, 64)
            multimask_output = False,
            )
        
        return(low_res_logits)
    
    def set_image(self, img_path): 
        # Original code: the ToCanonical function doesn't work without metadata anyway, so it efectively only reads in the image. For ease of preserving metadata, I use nib
        img = nib.load(img_path)
        img_data = img.get_fdata()
        self.img = img_data
        self.affine = img.affine
        self.image_set = True


    def clear_embeddings(self):
        self.stored_cropping_params, self.stored_padding_params, self.stored_patch_list = None, None, None

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
            coords = prompt.coords
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
            img3D_roi = img3D_roi.unsqueeze(dim=0).to(self.device)
            patch_embedding = self.model.image_encoder(img3D_roi.to(self.device)) # (1, 384, 16, 16, 16)

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

            window_list.append((patch_embedding, pos3D_roi))
        return cropping_params, padding_params, window_list

    def preprocess_prompt(self, pts_prompt, cropping_params, padding_params):

        coords = pts_prompt.coords
        labels = pts_prompt.labels

        # Transform prompt in line with image transforms
        point_offset = np.array([padding_params[0]-cropping_params[0], padding_params[2]-cropping_params[2], padding_params[4]-cropping_params[4]])
        coords = coords + point_offset
        
        batch_points = torch.from_numpy(coords).unsqueeze(0).to(self.device)
        batch_labels = torch.tensor(labels).unsqueeze(0).to(self.device)
        if self.use_only_first_point: # use only the first point since the model wasn't trained to receive multiple points in one go 
            batch_points = batch_points[:, :1]
            batch_labels = batch_labels[:, :1]
        
        return batch_points, batch_labels
    
    @torch.no_grad()
    def predict(self, prompt, prev_low_res_logits = None,
                cheat = False, gt = None, 
                store_patching = False, use_stored_patching = False,
                return_low_res_logits = False,
                transform = True): # If iterating, use previous patching, previous embeddings
        if not isinstance(prompt, SAMMed3DInferer.supported_prompts):
            raise ValueError(f'Unsupported prompt type: got {type(prompt)}')
        if not self.image_set:
            raise RuntimeError('Must first set image!')

        prompt.coords = prompt.coords[:,::-1] # Points are in xyz, but must be in zyx to align to image in row-major format.

        if use_stored_patching:
            if (self.stored_cropping_params is None) or (self.stored_padding_params is None) or (self.stored_patch_list is None):
                raise RuntimeError('No stored patchings to use!')
            cropping_params, padding_params, patch_list = self.stored_cropping_params, self.stored_padding_params, self.stored_patch_list
        else: # If stored patchings shouldn't be used, generate new ones
            cropping_params, padding_params, patch_list = self.preprocess_into_patches(self.img, prompt, cheat, gt)
        if store_patching and not use_stored_patching: # store patching if desired. If use_stored_patching, this would do nothing
            self.stored_cropping_params, self.stored_padding_params, self.stored_patch_list = cropping_params, padding_params, patch_list
            
        coords, labels = self.preprocess_prompt(prompt, cropping_params, padding_params)
        if use_stored_patching or cheat: # Check that the prompt lies within the patch
            if torch.any(torch.logical_or(coords<0, coords>=128)): 
                raise RuntimeError('Prompt coordinates do not lie within stored patch!')

        pred = np.zeros_like(self.img, dtype=np.uint8)
        for (patch_embedding, pos3D) in patch_list:
            if prev_low_res_logits is not None: # if previous low res logits are present, add number and channel dimensions
                prev_low_res_logits = prev_low_res_logits.unsqueeze(0).unsqueeze(0).to(self.device)
            else: # If no low res mask supplied, create one consisting of zeros
                low_res_spatial_shape = [dim//4 for dim in SAMMed3DInferer.required_shape] # batch and channel dimensions remain the same, spatial dimensions are quartered 
                prev_low_res_logits = torch.zeros([1,1] + low_res_spatial_shape).to(self.device) # [1,1] is batch and channel dimensions

            low_res_logits = self.segment(patch_embedding, prev_low_res_logits, coords, labels)
            logits = F.interpolate(low_res_logits, size = SAMMed3DInferer.required_shape, mode = 'trilinear', align_corners = False).detach().cpu().squeeze()
            seg_mask = (logits>0.5).numpy().astype(np.uint8)
            ori_roi, pred_roi = pos3D["ori_roi"], pos3D["pred_roi"]
            
            seg_mask_roi = seg_mask[..., pred_roi[0]:pred_roi[1], pred_roi[2]:pred_roi[3], pred_roi[4]:pred_roi[5]]
            pred[..., ori_roi[0]:ori_roi[1], ori_roi[2]:ori_roi[3], ori_roi[4]:ori_roi[5]] = seg_mask_roi

        if transform:
            pred = nib.Nifti1Image(pred, self.affine)
        
        if return_low_res_logits:
            return pred, low_res_logits.cpu().squeeze()
        else:
            return pred