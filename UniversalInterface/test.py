from utils.prompt import get_bbox3d
from utils.prompt_3d import get_pos_clicks3D
import numpy as np
import monai.transforms as transforms

checkpoint = "/home/t722s/Desktop/UniversalModels/TrainedModels/SegVol_v1.pth"
from importlib import reload
import utils.segvolClass as c
inferer = c.SegVolInferer(checkpoint)
#inferer = SegVolInferer(checkpoint)
