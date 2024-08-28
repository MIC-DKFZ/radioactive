from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Sequence
import numpy as np
from intrab.prompts.prompt import PromptStep

# Mask corresponds to being an interactive method allowing
prompt_categories = Literal["box", "point", "mask"]


class Inferer(ABC):
    supported_prompts: Sequence[prompt_categories] = ...
    dim: Literal[2, 3] = ...
    pass_prev_prompts: bool = ...

    def __init__(self, checkpoint_path, device):
        self.device = device
        self.model = self.load_model(checkpoint_path, device)
        self.loaded_image: Path | None = None
        # Just used for 2D models and not for 3D models
        self.image_embeddings_dict = {}

    @abstractmethod
    def load_model(self, checkpoint_path, device):
        """Load the model from a checkpoint"""
        pass

    @abstractmethod
    def preprocess_img(self, img: np.ndarray):
        """Any necessary preprocessing step like"""
        pass

    @abstractmethod
    def preprocess_prompt(self, prompt: PromptStep):
        pass

    # ToDo: Check if this should be an abstract method.
    #   Currently MedSAM did not have this.
    def postprocess_mask(self, mask):
        """Any necessary postprocessing steps"""
        pass

    @abstractmethod
    def predict(self, prompts: PromptStep):
        """Obtain logits"""
        pass

    def _image_already_loaded(self, img_path):
        return self.loaded_image is not None and img_path == self.loaded_image

    @abstractmethod
    def set_image(self, img_path):
        """Set the image to be segmented and make sure it adheres to local model requirements"""
        pass

    @abstractmethod
    def transform_to_model_coords(self, some_array: np.ndarray) -> np.ndarray:
        """Transform the coordinates to the model's coordinate system"""
        pass

    @abstractmethod
    def get_transformed_groundtruth(self, gt_path) -> np.ndarray:
        """Transform the groundtruth to the model's coordinate system"""
        pass
