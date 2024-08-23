from abc import ABC


class Inferer(ABC):
    def __init__(self, model):
        self.model = model

    def preprocess_img(self, img):
        """Any necessary preprocessing steps"""
        pass

    def preprocess_prompt(self, prompt):
        pass

    def postprocess_mask(self, mask):
        """Any necessary postprocessing steps"""
        pass

    def predict(self, model, inputs, device="cuda", keep_img_embedding=True):
        """Obtain logits"""
        pass
