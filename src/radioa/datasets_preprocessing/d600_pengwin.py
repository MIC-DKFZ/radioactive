import json
from pathlib import Path

import nrrd
import numpy as np
from os.path import join
from natsort import natsorted
import os
import SimpleITK as sitk
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

from tqdm import tqdm
from radioa.utils.paths import get_dataset_path
from tempfile import TemporaryDirectory
from toinstance import InstanceNrrd


# ToDo: The fractured pieces should probably be predicted as instances instead of semantic classes.
#


def load_filenames(load_dir, extension=None, return_path=False, return_extension=False):
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
                            filename = str(filename)[: -len(ext)]
            if return_path:
                filename = join(load_dir, filename)
            filepaths.append(filename)
    filepaths = np.asarray(filepaths)
    filepaths = natsorted(filepaths)

    return filepaths


@dataclass
class MedVol:
    array: Union[np.ndarray, str]
    spacing: Optional[np.ndarray] = None
    origin: Optional[np.ndarray] = None
    direction: Optional[np.ndarray] = None
    header: Optional[Dict] = None
    copy: Optional["MedVol"] = field(default=None, repr=False)

    def __post_init__(self):
        # Validate array: Must be a 3D array
        if not ((isinstance(self.array, np.ndarray) and self.array.ndim == 3) or isinstance(self.array, str)):
            raise ValueError("array must be a 3D numpy array or a filepath string")

        if isinstance(self.array, str):
            self._load(self.array)

        # Validate spacing: Must be None or a 1D array with three floats
        if self.spacing is not None:
            if not (
                isinstance(self.spacing, np.ndarray)
                and self.spacing.shape == (3,)
                and np.issubdtype(self.spacing.dtype, np.floating)
            ):
                raise ValueError("spacing must be None or a 1D numpy array with three floats")

        # Validate origin: Must be None or a 1D array with three floats
        if self.origin is not None:
            if not (
                isinstance(self.origin, np.ndarray)
                and self.origin.shape == (3,)
                and np.issubdtype(self.origin.dtype, np.floating)
            ):
                raise ValueError("origin must be None or a 1D numpy array with three floats")

        # Validate direction: Must be None or a 3x3 array of floats
        if self.direction is not None:
            if not (
                isinstance(self.direction, np.ndarray)
                and self.direction.shape == (3, 3)
                and np.issubdtype(self.direction.dtype, np.floating)
            ):
                raise ValueError("direction must be None or a 3x3 numpy array of floats")

        # Validate header: Must be None or a dictionary
        if self.header is not None and not isinstance(self.header, dict):
            raise ValueError("header must be None or a dictionary")

        # If copy is set, copy fields from the other Nifti instance
        if self.copy is not None:
            self._copy_fields_from(self.copy)

    def _copy_fields_from(self, other: "MedVol"):
        if self.spacing is None:
            self.spacing = other.spacing
        if self.origin is None:
            self.origin = other.origin
        if self.direction is None:
            self.direction = other.direction
        if self.header is None:
            self.header = other.header

    def _load(self, filepath):
        image_sitk = sitk.ReadImage(filepath)
        self.array = sitk.GetArrayFromImage(image_sitk)
        self.spacing = np.array(image_sitk.GetSpacing()[::-1])
        self.origin = np.array(image_sitk.GetOrigin())
        self.direction = np.array(image_sitk.GetDirection()).reshape(3, 3)
        self.header = {key: image_sitk.GetMetaData(key) for key in image_sitk.GetMetaDataKeys()}

    def save(self, filepath):
        image_sitk = sitk.GetImageFromArray(self.array)
        image_sitk.SetSpacing(self.spacing.tolist()[::-1])
        image_sitk.SetOrigin(self.origin.tolist())
        image_sitk.SetDirection(self.direction.flatten().tolist())
        for key, value in self.header.items():
            image_sitk.SetMetaData(key, value)
        sitk.WriteImage(image_sitk, filepath)


def preprocess(raw_data_path: Path):
    """Preprocess the Pengwin dataset to the final format."""
    train_images_p1 = raw_data_path / "PENGWIN_CT_train_images_part1"
    train_images_p2 = raw_data_path / "PENGWIN_CT_train_images_part2"
    train_labels = raw_data_path / "PENGWIN_CT_train_labels"

    train_images = list(train_images_p1.iterdir()) + list(train_images_p2.iterdir())
    train_labels = list(train_labels.iterdir())

    # ------------------------------ Convert Images ------------------------------ #
    target_image_dir = get_dataset_path() / "Dataset600_pengwin" / "imagesTr"
    target_image_dir.mkdir(parents=True, exist_ok=True)
    for im in tqdm(train_images, desc="Converting images"):
        target_image = target_image_dir / im.name.replace(".mha", "_0000.nii.gz")
        if not target_image.exists():
            vol = MedVol(str(im))
            vol.save(str(target_image))

    # ------------------------------ Convert Labels ------------------------------ #
    # The MedVol stuff maps connected components into one of 10 bins for each semantic classes.
    # We have 3 semantic classes but are safe, so we just upscale this.
    # In our eval we only care about bone fragments, and not by which bone they belong to.

    target_label_dir = get_dataset_path() / "Dataset600_pengwin" / "labelsTr"
    target_label_dir.mkdir(parents=True, exist_ok=True)
    for im in tqdm(train_labels, desc="Converting labels"):
        target_label = target_label_dir / im.name.replace(".mha", ".nii.gz")
        if not target_label.exists():
            vol = MedVol(str(im))
            with TemporaryDirectory() as tempdir:
                vol.save(str(Path(tempdir) / "temp.nii.gz"))
                img = sitk.ReadImage(str(Path(tempdir) / "temp.nii.gz"))
                # ------------- We just make instances numerically growing from 1 ------------ #
                img_array = sitk.GetArrayFromImage(img)
                for cnt, inst_id in enumerate(np.unique(img_array)):
                    if inst_id == 0:
                        continue
                    img_array[img_array == inst_id] = cnt
                # ------------------------------- Copy old info ------------------------------ #
                new_img = sitk.GetImageFromArray(img_array)
                new_img.CopyInformation(img)
                # ------------------------------- Save to disk ------------------------------- #
                sitk.WriteImage(new_img, target_label)

            # vol.save(str(target_label_dir / im.name.replace(".mha", ".nii.gz")))

    # ------------------------------- Dataset Json ------------------------------- #
    with open(get_dataset_path() / "Dataset600_pengwin" / "dataset.json", "w") as f:
        json.dump(
            {
                "channel_names": {"0": "CT"},
                "labels": {"background": 0, "bone_fragement": 1},
                "numTraining": 100,
                "file_ending": ".nii.gz",
                "name": "Pengwin_challenge_2024",
                "reference": "https://pengwin.grand-challenge.org/",
                "release": "https://doi.org/10.5281/zenodo.10927452",
                "overwrite_image_reader_writer": "NibabelIOWithReorient",
            },
            f,
        )
