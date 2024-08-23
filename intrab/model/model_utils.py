from intrab.model.SAM import SAMInferer
from intrab.model.SAMMed2D import SAMMed2DInferer
from intrab.model.MedSAM import MedSAMInferer
from intrab.model.SAMMed3D import SAMMed3DInferer
from intrab.model.segvol import SegVolInferer


inferer_registry_2d = {
    "sam": SAMInferer,
    "sammed2d": SAMMed2DInferer,
    "medsam": MedSAMInferer,
    "sammed3d": SAMMed3DInferer,
}


def sam_lazy(checkpoint_path, device):
    from intrab.model.SAM import SAMInferer

    return SAMInferer(checkpoint_path, device)


def sammed2d_lazy(checkpoint_path, device):
    return SAMMed2DInferer(checkpoint_path, device)


def medsam_lazy(checkpoint_path, device):

    return MedSAMInferer(checkpoint_path, device)


def segvol_lazy(checkpoint_path, device):

    return SegVolInferer(checkpoint_path, device)


def sammed3d_lazy(checkpoint_path, device):

    return SAMMed3DInferer(checkpoint_path, device)
