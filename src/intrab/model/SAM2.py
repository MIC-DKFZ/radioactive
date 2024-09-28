import torch
from intrab.model.SAM import SAMInferer
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2Inferer(SAMInferer):

    def load_model(self, checkpoint_path, device):
        sam2_model = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        return sam2_model
