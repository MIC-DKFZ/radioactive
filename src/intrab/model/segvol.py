from argparse import Namespace
from intrab.model.inferer import Inferer
from intrab.prompts.prompt import Boxes3D, Points
from intrab.utils.SegVol_segment_anything.network.model import SegVol
import torch
import os
import sys
from intrab.utils.SegVol_segment_anything.monai_inferers_utils import (
    build_binary_points,
    build_binary_cube,
    logits2roi_coor,
    sliding_window_inference,
)
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import monai.transforms as transforms
from copy import deepcopy

from intrab.utils.SegVol_segment_anything import sam_model_registry


class MinMaxNormalization(transforms.Transform):
    def __call__(self, data):
        d = dict(data)
        k = "image"
        d[k] = d[k] - d[k].min()
        d[k] = d[k] / np.clip(d[k].max(), a_min=1e-8, a_max=None)
        return d


class DimTranspose(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.swapaxes(d[key], -1, -3)
        return d


class ForegroundNormalization(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            d[key] = self.normalize(d[key])
        return d

    def normalize(self, ct_narray):
        ct_voxel_ndarray = ct_narray.copy()
        ct_voxel_ndarray = ct_voxel_ndarray.flatten()
        thred = np.mean(ct_voxel_ndarray)
        voxel_filtered = ct_voxel_ndarray[(ct_voxel_ndarray > thred)]
        upper_bound = np.percentile(voxel_filtered, 99.95)
        lower_bound = np.percentile(voxel_filtered, 00.05)
        mean = np.mean(voxel_filtered)
        std = np.std(voxel_filtered)
        ### transform ###
        ct_narray = np.clip(ct_narray, lower_bound, upper_bound)
        ct_narray = (ct_narray - mean) / max(std, 1e-8)
        return ct_narray


def load_segvol(checkpoint_path):
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Add this directory to the sys.path to allow relative path to clip checkpoint
    if script_directory not in sys.path:
        sys.path.append(script_directory)

    args = Namespace(
        test_mode=True,
        resume=checkpoint_path,
        infer_overlap=0.5,
        spatial_size=(32, 256, 256),
        patch_size=(4, 16, 16),
        clip_ckpt="./utils/SegVol_segment_anything/clip",  # This might not work if not running the .py in the base dir
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
        segvol_model.load_state_dict(checkpoint["model"], strict=True)
    segvol_model.eval()

    return segvol_model


class SegVolInferer(Inferer):

    pass_prev_prompts = True
    dim = 3
    supported_prompts = ("box", "point")

    def __init__(self, checkpoint_path, device="cuda"):
        if device != "cuda":
            raise RuntimeError("segvol can only be run on cuda.")
        self.model = load_segvol(checkpoint_path)
        self.prev_mask = None
        self.inputs = None
        self.mask_threshold = 0
        self.infer_overlap = 0.5
        self.verbose = True  # Not used anywhere, kept for alignment with 2d models.

        self.spatial_size = (32, 256, 256)
        self.transform = transforms.Compose(
            [
                transforms.Orientationd(
                    keys=["image"], axcodes="RAS"
                ),  # Doesn't actually do anything since metadata is discarded. Kept for comparability to original
                ForegroundNormalization(keys=["image"]),
                DimTranspose(keys=["image"]),
                MinMaxNormalization(),
                transforms.SpatialPadd(keys=["image"], spatial_size=self.spatial_size, mode="constant"),
                transforms.CropForegroundd(keys=["image"], source_key="image"),
                transforms.ToTensord(keys=["image"]),
            ]
        )
        self.zoom_out_transform = transforms.Resized(
            keys=["image"], spatial_size=self.spatial_size, mode="nearest-exact"
        )
        self.img_loader = transforms.LoadImage()

    def set_image(self, img_path):
        if self._image_already_loaded(img_path=img_path):
            return
        item = {}
        # generate ct_voxel_ndarray
        img, metadata = self.img_loader(img_path)
        self.affine = metadata["affine"]
        self.shape = img.shape
        img = np.expand_dims(
            img, axis=0
        )  # Ensure image is in CDWH (spatial dimensions will be assumed to be RAS anyway)
        item["image"] = img

        # transform
        item = self.transform(item)
        self.start_coord = item["foreground_start_coord"]  # Store metadata for later inveresion of transformations
        self.end_coord = item["foreground_end_coord"]

        item_zoom_out = self.zoom_out_transform(item)
        item["zoom_out_image"] = item_zoom_out["image"]
        image, image_zoom_out = item["image"].float().unsqueeze(0), item["zoom_out_image"].float().unsqueeze(0)
        image_single = image[0, 0]  #
        self.cropped_shape = image_single.shape
        self.img, self.img_zoom_out = image_single, image_zoom_out

    def preprocess_prompt(self, prompt, prompt_type, text_prompt=None):
        """
        Preprocessing steps:
            - Modify in line with the volume cropping
            - Modify in line with the interpolation
            - Collect into a dictionary of slice:slice prompt
        """
        # text_single, box_single, binary_cube_resize, points_single, binary_points_resize = prompt

        # Clean up
        box_single = points_single = binary_cube = binary_points = None

        if prompt_type == "point":
            # prompt.coords = prompt.coords[:,::-1]
            coords, labels = prompt.coords, prompt.labels
            # coords = coords[:,::-1]
            # Transform in line with cropping
            coords = coords - self.start_coord
            coords = np.maximum(coords, 0)
            coords = np.minimum(coords, self.end_coord)
            coords = np.round(
                coords * np.array(self.spatial_size) / np.array(self.cropped_shape)
            )  # Transform in line with resizing

            coords, labels = torch.from_numpy(coords).float(), torch.from_numpy(labels).float()
            points_single = (coords.unsqueeze(0).cuda(), labels.unsqueeze(0).cuda())
            binary_points_resize = build_binary_points(coords, labels, self.spatial_size)
            binary_points = F.interpolate(
                binary_points_resize.unsqueeze(0).unsqueeze(0).float(), size=self.cropped_shape, mode="nearest"
            )[0][0]
        if prompt_type == "box":
            # Transform box in line with image transformations
            box_single = np.array(prompt.bbox)  # Change from pair of points to 2x3 array
            box_single = box_single - self.start_coord  # transform in line with cropping
            box_single[0] = np.maximum(box_single[0], 0)
            box_single[1] = np.minimum(box_single[1], self.end_coord)
            box_single = np.round(
                box_single * np.array(self.spatial_size) / np.array(self.cropped_shape)
            )  # Transform in line with resizing
            box_single = box_single.flatten()
            box_single = torch.from_numpy(box_single).unsqueeze(0).float().cuda()

            # Obtain binary cube
            binary_cube_resize = build_binary_cube(box_single, self.spatial_size)
            binary_cube = F.interpolate(
                binary_cube_resize.unsqueeze(0).unsqueeze(0).float(), size=self.cropped_shape, mode="nearest"
            )[0][0]

        text_single = text_prompt
        prompt = text_single, box_single, binary_cube, points_single, binary_points
        return prompt

    @torch.no_grad()
    def segment(self, image_single, image_single_resize, prompt, prompt_type):
        # zoom-out inference:
        text_single, box_single, binary_cube, points_single, binary_points = prompt
        logits_global_zoom_out = self.model(
            image_single_resize.cuda(), text=text_single, boxes=box_single, points=points_single
        )

        # resize back global logits
        logits_global_zoom_out = F.interpolate(logits_global_zoom_out.cpu(), size=self.cropped_shape, mode="nearest")[
            0
        ][0]

        # zoom-in inference:
        min_d, min_h, min_w, max_d, max_h, max_w = logits2roi_coor(self.spatial_size, logits_global_zoom_out)
        if min_d is None:
            raise RuntimeError("Fail to detect foreground!")

        # Crop roi
        image_single_cropped = (
            image_single[min_d : max_d + 1, min_h : max_h + 1, min_w : max_w + 1].unsqueeze(0).unsqueeze(0)
        )
        global_preds = (
            torch.sigmoid(logits_global_zoom_out[min_d : max_d + 1, min_h : max_h + 1, min_w : max_w + 1]) > 0.5
        ).long()

        prompt_reflection = None
        if prompt_type == "box":
            binary_cube_cropped = binary_cube[min_d : max_d + 1, min_h : max_h + 1, min_w : max_w + 1]
            prompt_reflection = (
                binary_cube_cropped.unsqueeze(0).unsqueeze(0),
                global_preds.unsqueeze(0).unsqueeze(0),
            )
        if prompt_type == "point":
            binary_points_cropped = binary_points[min_d : max_d + 1, min_h : max_h + 1, min_w : max_w + 1]
            prompt_reflection = (
                binary_points_cropped.unsqueeze(0).unsqueeze(0),
                global_preds.unsqueeze(0).unsqueeze(0),
            )

        ## inference
        logits_single_cropped = sliding_window_inference(
            image_single_cropped.cuda(),
            prompt_reflection,
            self.spatial_size,
            1,
            self.model,
            self.infer_overlap,
            text=text_single,
            use_box=(prompt_type == "box"),
            use_point=(prompt_type == "point"),
        )
        logits_single_cropped = logits_single_cropped.cpu().squeeze()
        logits_global_zoom_in = logits_global_zoom_out.clone()
        logits_global_zoom_in[min_d : max_d + 1, min_h : max_h + 1, min_w : max_w + 1] = logits_single_cropped

        res = (image_single, points_single, box_single, logits_global_zoom_out, logits_global_zoom_in)

        return res

    def postprocess_mask(self, mask, return_logits):
        """
        Postprocessing steps:
            - TODO
        """
        if not return_logits:
            mask = (mask > 0).to(torch.uint8)  # Variable name says logits but it's actually raw results.
        # Invert transform
        mask = mask.transpose(-1, -3)
        start_coord, end_coord = deepcopy(self.start_coord), deepcopy(self.end_coord)
        start_coord[-1], start_coord[-3] = start_coord[-3], start_coord[-1]
        end_coord[-1], end_coord[-3] = end_coord[-3], end_coord[-1]
        segmentation = torch.zeros(self.shape)
        segmentation[start_coord[0] : end_coord[0], start_coord[1] : end_coord[1], start_coord[2] : end_coord[2]] = (
            mask
        )
        segmentation = segmentation.cpu().numpy().astype(np.uint8)  # Check if .cpu is necessary

        return segmentation

    def predict(self, prompt, text_prompt=None, return_logits=False, transform=True, seed=1):
        if self.loaded_image is None:
            raise RuntimeError("Must first set image!")

        if not isinstance(prompt, (Boxes3D, Points)):
            raise TypeError(
                "Prompts must be 3d bounding boxes or points, and must be supplied as an instance of Boxes3D or Points"
            )

        prompt_type = "box" if isinstance(prompt, Boxes3D) else "point"

        if prompt_type == "point":
            torch.manual_seed(
                seed
            )  # New points are sampled in the zoom-in section of zoom-out zoom-in inference leaving some randomness even after a point prompt is fixed.

        image_single, image_single_resize = self.img, self.img_zoom_out

        prompt = deepcopy(prompt)
        prompt = self.preprocess_prompt(prompt, prompt_type, text_prompt)

        res = self.segment(image_single, image_single_resize, prompt, prompt_type)

        segmentation = self.postprocess_mask(res[-1], return_logits)

        if transform:
            segmentation = nib.Nifti1Image(segmentation, self.affine)

        return segmentation
