from argparse import Namespace
import torch

from .SAMMed3D_segment_anything.build_sam import sam_model_registry as registry_sam
from .MedSAM_segment_anything import sam_model_registry as registry_medsam
from .SAMMed2D_segment_anything import sam_model_registry as registry_sammed2d
from .SAMMed3D_segment_anything.build_sam3D import build_sam3D_vit_b_ori

from classes.SAMClass import SAMWrapper, SAMInferer
from classes.SAMMed2DClass import SAMMed2DInferer
from classes.MedSAMClass import MedSAMInferer
from classes.SAMMed3DClass import SAMMed3DInferer

inferer_registry = {
    'sam': SAMInferer,
    'sammed2d': SAMMed2DInferer,
    'medsam': MedSAMInferer,
    'sammed3d': SAMMed3DInferer
}

def load_sam(checkpoint_path, device = 'cuda', image_size = 1024):
    args = Namespace()
    args.image_size = image_size
    args.sam_checkpoint = checkpoint_path
    args.model_type = 'vit_h'
    model = registry_sam[args.model_type](args).to(device)
    return(model)

def load_medsam(checkpoint_path, device = 'cuda'):
    medsam_model = registry_medsam['vit_b'](checkpoint=checkpoint_path)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    return(medsam_model)

def load_sammed2d(checkpoint_path, device = 'cuda'):
    args = Namespace()
    args.image_size = 256
    args.encoder_adapter = True
    args.sam_checkpoint = checkpoint_path
    model = registry_sammed2d["vit_b"](args).to(device)

    return(model)

def load_sammed3d(checkpoint_path, device = 'cuda'):

    sam_model_tune = build_sam3D_vit_b_ori(checkpoint=None)
    if checkpoint_path is not None:
        model_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = model_dict['model_state_dict']
        sam_model_tune.load_state_dict(state_dict)
        sam_model_tune.to(device)

    return (sam_model_tune)

def load_sam_inferer(checkpoint_path, device = 'cuda', image_size = 1024):
    args = Namespace()
    args.image_size = image_size
    args.sam_checkpoint = checkpoint_path
    args.model_type = 'vit_h'
    model = registry_sam[args.model_type](args).to(device)
    sam_wrapper = SAMWrapper(model, device)
    sam_inferer = SAMInferer(sam_wrapper)
    return(sam_inferer)