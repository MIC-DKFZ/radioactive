from abc import ABC, abstractmethod


class Prompt(): # Abstract class for prompts to be inputted with the SAM adjusted model
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f'{self.name} prompt({self.value})'

class Points(Prompt):
    def __init__(self, value):
        super().__init__(name = 'points', value = value)

    def get_slices_to_infer(self):
        unique_zs = set(self.value['coords'][:,2].numpy())
        return(unique_zs)

class Boxes2d(Prompt):
    def __init__(self, value):
        '''
        self.value must be a dictionary with items {slice_number:bounding box for slice}
        '''
        self.name = 'boundingBox'
        self.value = value

    def get_slices_to_infer(self):
        return(list(self.value.keys()))
        
class Inferer(ABC):
    def __init__(self, model):
        self.model = model

    def preprocess_img(self, img):
        '''Any necessary preprocessing steps'''
        pass

    def postprocess_mask(self, mask):
        '''Any necessary postprocessing steps'''
        pass
 
    def predict(self, model, inputs, device = 'cuda', keep_img_embedding = True):
        '''Obtain logits '''
        pass

class SegmenterWrapper(ABC):
    supported_prompts: list # List of supported prompt types

    @abstractmethod
    def __init__(self, model, device: str):
        '''
        Initialise with the usual model and device (eg cuda)
        '''

    @abstractmethod
    def __call__(self, img, prompt):
        '''
        Take only the image and prompt and return a 2D/3D logit mask as appropriate
        '''
        if prompt.is_point():
            pass
        else:
            raise ValueError("This model does not support these")
