from argparse import Namespace
from pathlib import Path
from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib
from pathlib import Path
from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib
from radioa.model.inferer import Inferer
from radioa.model.segvol import SegVolInferer
from radioa.prompts.prompt import Boxes3D, Points, PromptStep
from radioa.utils.SegVol_segment_anything.network.model import SegVol
import torch
import os
import sys

from radioa.utils.SegVol_segment_anything import sam_model_registry
from radioa.utils.paths import get_model_path

import numpy as np


def load_segvol(checkpoint_path):
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Add this directory to the sys.path to allow relative path to clip checkpoint
    if script_directory not in sys.path:
        sys.path.append(script_directory)

    model_ckpt = get_model_path() / "SegVol_v1.pth"
    args = Namespace(
        test_mode=True,
        resume=checkpoint_path,
        infer_overlap=0.5,
        spatial_size=(32, 256, 256),
        patch_size=(4, 16, 16),
        clip_ckpt=str(model_ckpt),  # This might not work if not running the .py in the base dir
    )

    gpu = 0

    sam_model = sam_model_registry["vit"](args=args)

    segvol_model = SegVol(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        clip_ckpt=args.clip_ckpt,
        roi_size=args.spatial_size,
        patch_size=args.patch_size,
        test_mode=args.test_mode,
    ).cuda()
    segvol_model = torch.nn.DataParallel(segvol_model, device_ids=[gpu])

    if os.path.isfile(args.resume):
        ## Map model to be loaded to specified single GPU
        loc = "cuda:{}".format(gpu)
        checkpoint = torch.load(args.resume, map_location=loc)
        segvol_model.load_state_dict(checkpoint["model"], strict=False)
    segvol_model.eval()

    return segvol_model


class SegVolNormInferer(SegVolInferer):

    def set_image(self, img_path):
        if self._image_already_loaded(img_path=img_path):
            return
        img_nib = load_any_to_nib(img_path)
        self.orig_affine = img_nib.affine
        self.orig_shape = img_nib.shape

        self.img, self.img_zoom_out, self.start_coord, self.end_coord = self.transform_to_model_coords_dense(
            img_nib, is_seg=False
        )
        self.cropped_shape = self.img.shape
        self.loaded_image = img_path

        # clip image to 0.5% - 99.5%
        self.global_min = np.percentile(self.img, 0.5)
        self.global_max = np.percentile(self.img, 99.5)

        # Clip the image array
        self.img = np.clip(self.img, self.global_min, self.global_max)
        self.img = torch.as_tensor(self.img)
