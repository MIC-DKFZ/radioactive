from argparse import Namespace
from pathlib import Path
from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib
from pathlib import Path
from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib
from radioa.model.inferer import Inferer
from radioa.prompts.prompt import Boxes3D, Points, PromptStep
from radioa.utils.SegVol_segment_anything.network.model import SegVol
import torch
import os
import sys
from radioa.utils.SegVol_segment_anything.monai_inferers_utils import (
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

from radioa.utils.SegVol_segment_anything import sam_model_registry
from radioa.utils.paths import get_model_path
from radioa.utils.transforms import resample_to_shape_sparse


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
        ct_voxel_ndarray = deepcopy(ct_narray)
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


class SegVolInferer(Inferer):

    pass_prev_prompts = True
    dim = 3
    supported_prompts = ("box", "point")

    def __init__(self, checkpoint_path, device="cuda"):
        if device != "cuda":
            raise RuntimeError("segvol can only be run on cuda.")
        super().__init__(checkpoint_path, device)
        self.prev_mask = None
        self.inputs = None
        self.mask_threshold = 0
        self.infer_overlap = 0.5
        self.start_coord = None
        self.end_coord = None

        self.spatial_size = (32, 256, 256)

        self.transform_img = transforms.Compose(
            [
                transforms.Orientationd(
                    keys=["image"], axcodes="RAS"
                ),  # Doesn't actually do anything since metadata is discarded. Kept for comparability to original
                ForegroundNormalization(keys=["image"]),
                DimTranspose(keys=["image"]),
                MinMaxNormalization(),
                transforms.SpatialPadd(keys=["image"], spatial_size=(32, 256, 256), mode="constant"),
                transforms.CropForegroundd(keys=["image"], source_key="image"),
                transforms.ToTensord(keys=["image"]),
            ]
        )

        self.transform_seg = transforms.Compose(
            [
                transforms.Orientationd(
                    keys=["image", "seg"], axcodes="RAS"
                ),  # Doesn't actually do anything since metadata is discarded. Kept for comparability to original
                DimTranspose(keys=["image", "seg"]),
                transforms.SpatialPadd(keys=["image", "seg"], spatial_size=(32, 256, 256), mode="constant"),
                transforms.CropForegroundd(keys=["image", "seg"], source_key="image"),
                transforms.ToTensord(keys=["image", "seg"]),
            ]
        )
        self.zoom_out_transform = transforms.Resized(
            keys=["image"], spatial_size=self.spatial_size, mode="nearest-exact"
        )
        self.img_loader = transforms.LoadImage()

    def load_model(self, checkpoint_path, device):
        return load_segvol(checkpoint_path)

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

    def transform_to_model_coords_dense(
        self, nifti: str | Path | nib.Nifti1Image, is_seg: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(nifti, (str, Path)):
            nifti = load_any_to_nib(nifti)

        if is_seg:
            # raise RuntimeError('An image must first be set before segmentations can be transformed')
            return np.zeros((10, 10, 10))  # Disgusting fix since groundtruth_model isn't needed

        item = {}
        # generate ct_voxel_ndarray
        img = nifti.get_fdata()
        img = np.expand_dims(
            img, axis=0
        )  # Ensure image is in CDWH (spatial dimensions will be assumed to be RAS anyway)

        # transform
        if not is_seg:
            item["image"] = img
            item = self.transform_img(item)
        else:
            dummy_img = np.zeros_like(self.img)
            dummy_img[
                self.start_coord[0] : self.end_coord[0],
                self.start_coord[1] : self.end_coord[1],
                self.start_coord[2] : self.end_coord[2],
            ] = 1  # used for crop transform; must crop the same way as the image was cropped.
            item["image"] = dummy_img
            item["seg"] = img

            item = self.transform_seg(item)
            item["image"] = item["seg"]
            del item["seg"]
            # now make appropriate for zoom out transform

        start_coord = item["foreground_start_coord"]  # Store metadata for later inveresion of transformations
        end_coord = item["foreground_end_coord"]

        item_zoom_out = self.zoom_out_transform(item)
        item["zoom_out_image"] = item_zoom_out["image"]
        image, image_zoom_out = item["image"].float().unsqueeze(0), item["zoom_out_image"].float().unsqueeze(0)
        image_single = image[0, 0]

        img, img_zoom_out = image_single, image_zoom_out
        return img, img_zoom_out, start_coord, end_coord

    def transform_to_model_coords_sparse(self, coords: np.ndarray) -> np.ndarray:
        # OrientationD: Do nothing

        # DimTranspose
        coords = coords[[2, 1, 0]]  # Swap the first and last coordinates

        # SpatialPad
        # Calculate padding amounts
        shape_after_dimtranspose = self.orig_shape[::-1]

        # Calculate padding needed for each axis
        total_pads = np.maximum(np.array(self.spatial_size) - np.array(shape_after_dimtranspose), 0)
        pad_starts = total_pads // 2

        # Adjust point coordinates for padding
        coords = coords + pad_starts

        # CropForegroundd
        coords = coords - self.start_coord

        # Ensure that coords lie within model patch - can occur if 'background' is segmented as gt
        coords = np.minimum(coords, self.cropped_shape)
        coords = np.maximum(coords, 0)

        # 5. Resized (if applicable)
        zoomed_out_coords = resample_to_shape_sparse(coords, self.cropped_shape, self.spatial_size, round=True)

        return zoomed_out_coords

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
            coords, labels = prompt.coords, prompt.labels
            # # coords = coords[:,::-1]
            # # Transform in line with cropping # These transforms were moved to transform_to_model_coordinates_sparse
            # coords = coords - self.start_coord
            # coords = np.maximum(coords, 0)
            # coords = np.minimum(coords, self.end_coord)
            # coords = np.round(
            #     coords * np.array(self.spatial_size) / np.array(self.cropped_shape)
            # )  # Transform in line with resizing

            coords, labels = torch.from_numpy(coords).float(), torch.from_numpy(labels).float()
            points_single = (coords.unsqueeze(0).cuda(), labels.unsqueeze(0).cuda())
            binary_points_resize = build_binary_points(coords, labels, self.spatial_size)
            binary_points = F.interpolate(
                binary_points_resize.unsqueeze(0).unsqueeze(0).float(), size=self.cropped_shape, mode="nearest"
            )[0][0]
        if prompt_type == "box":
            # # Transform box in line with image transformations # These transforms were moved to transform_to_model_coordinates_sparse
            # box_single = np.array(prompt.bbox)  # Change from pair of points to 2x3 array
            # box_single = box_single - self.start_coord  # transform in line with cropping
            # box_single[0] = np.maximum(box_single[0], 0)
            # box_single[1] = np.minimum(box_single[1], self.end_coord)
            # box_single = np.round(
            #     box_single * np.array(self.spatial_size) / np.array(self.cropped_shape)
            # )  # Transform in line with resizing

            box_single = np.array(prompt.bbox)  # Change from pair of points to 2x3 array
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

    def preprocess_img(self, img: np.ndarray):  # image preprocessing is handled in set_image
        pass

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
            res = (None, None, None, None, logits_global_zoom_out)
            return res
            # raise RuntimeError("Fail to detect foreground!")

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
            mask = (mask > 0).cpu().numpy().astype(np.uint8)

        return mask

    def inv_trans_dense(self, mask: np.ndarray) -> nib.Nifti1Image:
        # Invert transform
        shape_after_dimtranspose = self.orig_shape[::-1]
        # undo crop foreground
        padded_segmenation_shape = np.maximum(self.spatial_size, shape_after_dimtranspose)
        padded_segmentation = np.zeros(
            padded_segmenation_shape, dtype=np.uint8
        )  # Should be created in 32,256,256 shape
        padded_segmentation[
            self.start_coord[0] : self.end_coord[0],
            self.start_coord[1] : self.end_coord[1],
            self.start_coord[2] : self.end_coord[2],
        ] = mask  #  stick into 32,256,256 shape then undo pad # ToDo needs fixing, try MS instances:

        # undo spatial pad
        total_pads = np.maximum(np.array(self.spatial_size) - np.array(shape_after_dimtranspose), 0)
        pad_starts = total_pads // 2

        transposed_segmentation = padded_segmentation[
            pad_starts[0] : shape_after_dimtranspose[0] + pad_starts[0],
            pad_starts[1] : shape_after_dimtranspose[1] + pad_starts[1],
            pad_starts[2] : shape_after_dimtranspose[2] + pad_starts[2],
        ]

        # undo dim_transpose
        segmentation = np.transpose(transposed_segmentation, (2, 1, 0))

        segmentation = nib.Nifti1Image(segmentation, self.orig_affine)

        return segmentation

    def predict(
        self,
        prompt: PromptStep | Boxes3D,
        text_prompt=None,
        return_logits=False,
        prev_seg=None,
        promptstep_in_model_coord_system=False,
        seed=1,
    ):
        if self.loaded_image is None:
            raise RuntimeError("Must first set image!")

        if not isinstance(prompt, (Boxes3D, PromptStep)):
            raise TypeError(
                "Prompts must be 3d bounding boxes or points, and must be supplied as an instance of Boxes3D or PromptStep"
            )

        prompt_type = "box" if isinstance(prompt, Boxes3D) else "point"

        if prompt_type == "point":
            torch.manual_seed(
                seed
            )  # New points are sampled in the zoom-in section of zoom-out zoom-in inference leaving some randomness even after a point prompt is fixed.

        image_single, image_single_resize = self.img, self.img_zoom_out

        prompt = deepcopy(prompt)
        if not promptstep_in_model_coord_system:
            prompt = self.transform_promptstep_to_model_coords(prompt)
        prompt = self.preprocess_prompt(prompt, prompt_type, text_prompt)

        res = self.segment(image_single, image_single_resize, prompt, prompt_type)

        segmentation = self.postprocess_mask(res[-1], return_logits)

        segmentation_model_arr = segmentation

        # Turn into Nifti object in original space
        segmentation_orig_nib = self.inv_trans_dense(segmentation)

        low_res_logits = None  # low_res_logits aren't easily accessed nor used since segvol isn't interactive

        return segmentation_orig_nib, low_res_logits, segmentation_model_arr
