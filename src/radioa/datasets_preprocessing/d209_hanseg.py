import json
import math
import os
from pathlib import Path
import nrrd
import numpy as np
import SimpleITK as sitk
from typing import List

from tqdm import tqdm

from radioa.utils.paths import get_dataset_path
from tempfile import TemporaryDirectory
import re

join = os.path.join

HANSEG_LABEL_ORDER = {
    "background": 0,
    "A_Carotid_L": 1,
    "A_Carotid_R": 2,
    "Arytenoid": 3,
    "Bone_Mandible": 4,
    "Brainstem": 5,
    "BuccalMucosa": 6,
    "Cavity_Oral": 7,
    "Cochlea_L": 8,
    "Cochlea_R": 9,
    "Cricopharyngeus": 10,
    "Esophagus_S": 11,
    "Eye_AL": 12,
    "Eye_AR": 13,
    "Eye_PL": 14,
    "Eye_PR": 15,
    "Glnd_Lacrimal_L": 16,
    "Glnd_Lacrimal_R": 17,
    "Glnd_Submand_L": 18,
    "Glnd_Submand_R": 19,
    "Glnd_Thyroid": 20,
    "Glottis": 21,
    "Larynx_SG": 22,
    "Lips": 23,
    "OpticChiasm": 24,
    "OpticNrv_L": 25,
    "OpticNrv_R": 26,
    "Parotid_L": 27,
    "Parotid_R": 28,
    "Pituitary": 29,
    "SpinalCord": 30,
}


def subdirs(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [
        l(folder, i)
        for i in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, i))
        and (prefix is None or i.startswith(prefix))
        and (suffix is None or i.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res


def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def paddingSitkImage(im_sitk, target_shape, pad_value, output_format="sitk", verbose=False):
    """The function is zero-padding the image for a given size
    Parameters:
        - im_sitk: input image as an SitkImage
        - target_shape: list or tuple, new shape in the order of z(transversal plane), y, x
        - output_format: string, type of output: "sitk", "np"
        - verbose: bool, set it True if you want to print information about the padding
        - im_sitk_pad: SitkImage, the output image
    """
    # Calculate padding on both sides of the image
    im_nparr = sitk.GetArrayFromImage(im_sitk)
    bz, by, bx = (np.subtract(target_shape, im_nparr.shape)) / 2
    if bz < 0 or by < 0 or bx < 0:
        print(bz, by, bx)
        print("ERROR: One of the target dimensions is smaller than the original image, consider cropping")
        return
    bzh, byh, bxh = np.ceil((bz, by, bx)).astype(int)
    bzl, byl, bxl = np.floor((bz, by, bx)).astype(int)

    # Padding:
    im_nparr_pad = np.pad(im_nparr, ((bzh, bzl), (byh, byl), (bxh, bxl)), "constant", constant_values=pad_value)
    if verbose:
        print("PADDING:")
        print("Input Shape: ", im_nparr.shape)
        print("Padding - z: {:3d} + {:3d}".format(bzl, bzh))
        print("        - y: {:3d} + {:3d}".format(byl, byh))
        print("        - x: {:3d} + {:3d}".format(bxl, bxh))
        print("New Shape:   ", im_nparr_pad.shape)
        print()

    if output_format == "sitk":
        # Retransform nparray to SitkImage
        im_sitk_pad = sitk.GetImageFromArray(im_nparr_pad)
        im_sitk_pad.SetOrigin(im_sitk.GetOrigin())
        im_sitk_pad.SetSpacing(im_sitk.GetSpacing())
        im_sitk_pad.SetDirection(im_sitk.GetDirection())
        return im_sitk_pad
    elif output_format == "np":
        return im_nparr_pad
    else:
        print("ERROR: Unknown output format")
        return


def respaceSitkImage(
    image, new_spacing=[0.3125, 0.3125, 3.0], ignore_zratio=False, interpolation="nearest", verbose=False
):
    """The function respaces the image with a given spacing and interpolation
    Parameters:
        - image: input image as an SitkImage
        - new_spacing: list, new spacing in the order of x, y, z(transversal plane)
        - ignore_zratio: bool, True if you dont want to get interpolation error in the z-axis
                         due to the negligibly small deviation in the z spacing
        - interpolation: string, type of interpolation: "bspline", "linear", "nearest"
        - verbose: bool, True if you want to print information about the respacing steps
    """

    # Calculating the ratio of the old and the new spacing:
    orig_spacing = image.GetSpacing()
    if ignore_zratio:
        overwrite_zspacing = new_spacing[2]
        new_spacing[2] = image.GetSpacing()[2]
    spacing_ratio = np.divide(orig_spacing, new_spacing)
    if verbose:
        print("SPACING RATIO:")
        print(
            "Original Spacing: {:7.5f}, {:5.5f}, {:10.8f}".format(orig_spacing[0], orig_spacing[1], orig_spacing[2])
        )
        print("New Spacing:      {:6.5f}, {:5.5f}, {:10.8f}".format(new_spacing[0], new_spacing[1], new_spacing[2]))
        print(
            "Spacing Ratio:    {:6.5f}, {:5.5f}, {:10.8f}".format(
                spacing_ratio[0], spacing_ratio[1], spacing_ratio[2]
            )
        )
        print()

    # Calculating the new image size based on the spacing ratio
    orig_size = image.GetSize()
    new_size = orig_size * spacing_ratio
    for i in range(len(new_spacing)):
        if (new_size[i] - int(new_size[i])) < 0.5:
            new_size[i] = math.floor(new_size[i]) - 1
        else:
            new_size[i] = math.ceil(new_size[i]) - 1
    new_size = [int(s) for s in new_size]
    if verbose:
        print("IMAGE SIZES:")
        print("Original Size:    {:3d}, {:3d}, {:2d}".format(orig_size[0], orig_size[1], orig_size[2]))
        print("New Size:         {:3d}, {:3d}, {:2d}".format(new_size[0], new_size[1], new_size[2]))
        print()

    # Setting up the SITK Resampler and execute it
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    if interpolation == "bspline":
        resampler.SetInterpolator(sitk.sitkBSpline)
    elif interpolation == "linear":
        resampler.SetInterpolator(sitk.sitkLinear)
    elif interpolation == "nearest":
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        print("ERROR: Unknown interpolation method")
        return

    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampled_image = resampler.Execute(image)
    if ignore_zratio:
        resampled_image.SetSpacing([new_spacing[0], new_spacing[1], overwrite_zspacing])

    if verbose:
        print("FINAL SPACING:")
        print("New Spacing:     ", resampled_image.GetSpacing())
        print()

    return resampled_image


def move_image(moving_image: sitk.Image, move_map: sitk.ParameterMap, is_label: bool = False) -> sitk.Image:
    """Apply the transformation map to a moving image."""
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetMovingImage(moving_image)
    transformixImageFilter.SetTransformParameterMap(move_map)

    if is_label:
        transformixImageFilter.SetTransformParameter("ResampleInterpolator", "FinalNearestNeighborInterpolator")

    # Execute the transformation
    transformixImageFilter.Execute()

    # Get and return the transformed image
    return transformixImageFilter.GetResultImage()


def get_registration_map(fixed_image: sitk.Image, moving_image: sitk.Image) -> sitk.ParameterMap:
    """Get the transformation map by registering the moving image (CT) to the fixed image (MR)."""
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed_image)  # Fixed image is the MR image
    elastixImageFilter.SetMovingImage(moving_image)  # Moving image is the CT image

    # Set registration parameters (rigid transformation)
    parameterMap = sitk.GetDefaultParameterMap("rigid")
    parameterMap["AutomaticTransformInitialization"] = ["true"]
    parameterMap["AutomaticTransformInitializationMethod"] = ["CenterOfGravity"]

    elastixImageFilter.SetParameterMap(parameterMap)

    # Execute the registration
    elastixImageFilter.Execute()

    # Return the computed transformation parameter map
    return elastixImageFilter.GetTransformParameterMap()


def convert_HanSeg(inputfolder: Path, outputfolder: Path):
    case_dirs = list(inputfolder.glob("case_*"))
    output_image_dir = outputfolder / "imagesTr"
    output_label_dir = outputfolder / "labelsTr"
    output_image_dir.mkdir(exist_ok=True, parents=True)
    output_label_dir.mkdir(exist_ok=True, parents=True)

    print([c.name for c in case_dirs])
    for case_dir in tqdm(case_dirs, desc="Converting HaN-Seg"):
        print(f"Processing {case_dir.name}")
        ct_img_path = list(case_dir.rglob("*IMG_CT.nrrd"))[0]
        mr_img_path = list(case_dir.rglob("*IMG_MR_T1.nrrd"))[0]
        gt_paths = list(case_dir.rglob("*OAR*.nrrd"))
        # ------------------------ Assign each file a label id ----------------------- #
        label_wise_gt_paths: dict[int, Path] = {}
        for gt_p in gt_paths:
            class_name = re.sub(r".*_OAR_|\.seg\.nrrd", "", gt_p.name)
            hanseg_label_id: int = HANSEG_LABEL_ORDER[class_name]
            label_wise_gt_paths[hanseg_label_id] = gt_p

        # Read the CT and MR images
        ct_img = sitk.ReadImage(ct_img_path)
        im_MR = sitk.ReadImage(mr_img_path)

        sitk.WriteImage(im_MR, output_image_dir / (mr_img_path.name.replace("_IMG_MR_T1.nrrd", "_0000.nrrd")))

        # Preprocess MR to match the size and spacing of CT
        im_MR.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        im_ct_res = respaceSitkImage(
            image=im_MR, new_spacing=im_MR.GetSpacing(), ignore_zratio=False, interpolation="nearest", verbose=True
        )

        # Pad MR to match CT dimensions
        # im_MR_respad = paddingSitkImage(
        #     im_sitk=im_MR_res,
        #     target_shape=ct_img.GetSize()[::-1],
        #     pad_value=im_MR.GetPixel((0, 0, 0)),
        #     output_format="sitk",
        #     verbose=True,
        # )
        im_ct_res.SetOrigin(im_MR.GetOrigin())

        # Obtain the registration map (transformation) by registering CT to MR
        registration_map = get_registration_map(fixed_image=im_MR, moving_image=ct_img)

        moved_ct = move_image(moving_image=ct_img, move_map=registration_map)
        sitk.WriteImage(moved_ct, outputfolder / ct_img_path.name)
        # Apply the registration transformation to the CT image
        with TemporaryDirectory(dir="/dev/shm") as temp_dir:
            tmp_sitk_img_dirs = {}
            for label_id, gt_path in label_wise_gt_paths.items():
                lbl_img = sitk.ReadImage(gt_path)
                tmp_move_image = move_image(moving_image=lbl_img, move_map=registration_map, is_label=True)
                tmp_img_path = Path(temp_dir) / (gt_path.name.split(".")[0] + ".nrrd")
                sitk.WriteImage(tmp_move_image, tmp_img_path)
                tmp_sitk_img_dirs[label_id] = tmp_img_path
            arr, header = nrrd.read(list(tmp_sitk_img_dirs.values())[0])
            new_arr = np.zeros_like(arr)
            for label_id, gt_path in tmp_sitk_img_dirs.items():
                arr, _ = nrrd.read(gt_path)
                new_arr[arr == 1] = label_id
            header["encoding"] = "gzip"
            nrrd.write(str(output_label_dir / (case_dir.name + ".nrrd")), new_arr, header)
    # ------------------------------- Dataset Json ------------------------------- #
    with open(outputfolder / "dataset.json", "w") as f:
        json.dump(
            {
                "channel_names": {"0": "T1 MRI"},
                "labels": HANSEG_LABEL_ORDER,
                "numTraining": len(case_dirs),
                "file_ending": ".nrrd",
                "name": "HanSeg Challenge ",
                "reference": "https://han-seg2023.grand-challenge.org/han-seg2023/",
                "release": "https://zenodo.org/records/7442914#.ZBtfBHbMJaQ",
            },
            f,
        )


def preprocess(download_folder: Path):
    target_folder = get_dataset_path() / "Dataset209_hanseg_mr_oar"

    actual_download_dir = download_folder / "HaN-Seg" / "HaN-Seg" / "set_1"
    convert_HanSeg(actual_download_dir, target_folder)


def main():

    pass


if __name__ == "__main__":
    main()
