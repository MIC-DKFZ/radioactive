import numpy as np
from os.path import join
#from natsort import natsorted
import os
from PIL import Image
import SimpleITK as sitk


def load_filepaths(load_dir, extension=None, return_path=True, return_extension=True):
    filepaths = []
    if isinstance(extension, str):
        extension = tuple([extension])
    elif isinstance(extension, list):
        extension = tuple(extension)
    elif extension is not None and not isinstance(extension, tuple):
        raise RuntimeError("Unknown type for argument extension.")

    if extension is not None:
        extension = list(extension)
        for i in range(len(extension)):
            if extension[i][0] != ".":
                extension[i] = "." + extension[i]
        extension = tuple(extension)

    for filename in os.listdir(load_dir):
        if extension is None or str(filename).endswith(extension):
            if not return_extension:
                if extension is None:
                    filename = filename.split(".")[0]
                else:
                    for ext in extension:
                        if str(filename).endswith((ext)):
                            filename = str(filename)[:-len(ext)]
            if return_path:
                filename = join(load_dir, filename)
            filepaths.append(filename)
    filepaths = np.asarray(filepaths)
    filepaths = natsorted(filepaths)

    return filepaths


def load_image(filepath, return_meta=False):
    if filepath.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
        img = Image.open(filepath)
        img = np.array(img)
        metadata = None
    elif filepath.lower().endswith((".nii.gz", ".nii")):
        img, spacing, affine, header = load_nifti(filepath, return_meta=True)
        metadata = {"spacing": spacing, "affine": affine, "header": header}
    else:
        raise RuntimeError("Unsupported file format.")
    
    if return_meta:
        return img, metadata
    else:
        return img


def save_image(filepath, img, metadata=None):
    if filepath.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
        img = Image.fromarray(img)
        img.save(filepath)
    elif filepath.lower().endswith((".nii.gz", ".nii")):
        spacing, affine, header = None, None, None
        if metadata is not None:
            spacing, affine, header = metadata["spacing"], metadata["affine"], metadata["header"]
        save_nifti(filepath, img, spacing, affine, header)
    else:
        raise RuntimeError("Unsupported file format.")


def save_nifti(filename, image, spacing=None, affine=None, header=None, is_seg=False, dtype=None):
    if is_seg:
        image = np.rint(image)
        if dtype is None:
            image = image.astype(np.int16)  # In special cases segmentations can contain negative labels, so no np.uint8 by default

    if dtype is not None:
        image = image.astype(dtype)

    image = sitk.GetImageFromArray(image)

    if header is not None:
        [image.SetMetaData(key, header[key]) for key in header.keys()]

    if spacing is not None:
        image.SetSpacing(spacing)

    if affine is not None:
        pass  # How do I set the affine transform with SimpleITK? With NiBabel it is just nib.Nifti1Image(img, affine=affine, header=header)

    sitk.WriteImage(image, filename)



def load_nifti(filename, return_meta=False, is_seg=False):
    image = sitk.ReadImage(filename)
    image_np = sitk.GetArrayFromImage(image)

    if is_seg:
        image_np = np.rint(image_np)
        # image_np = image_np.astype(np.int16)  # In special cases segmentations can contain negative labels, so no np.uint8

    if not return_meta:
        return image_np
    else:
        spacing = image.GetSpacing()
        keys = image.GetMetaDataKeys()
        header = {key:image.GetMetaData(key) for key in keys}
        affine = None  # How do I get the affine transform with SimpleITK? With NiBabel it is just image.affine
        return image_np, spacing, affine, header



def normalize(x, source_limits=None, target_limits=None):
    if source_limits is None:
        source_limits = (x.min(), x.max())

    if target_limits is None:
        target_limits = (0, 1)

    if source_limits[0] == source_limits[1] or target_limits[0] == target_limits[1]:
        return x * 0
    else:
        x_std = (x - source_limits[0]) / (source_limits[1] - source_limits[0])
        x_scaled = x_std * (target_limits[1] - target_limits[0]) + target_limits[0]
        return x_scaled