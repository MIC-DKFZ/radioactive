import json
from pathlib import Path
import shutil
from typing import Literal
from loguru import logger
import nibabel as nib
import numpy as np
import requests
from toinstance import InstanceNrrd
import nrrd
import tempfile
import SimpleITK as sitk
from tqdm import tqdm
import gdown

import os

try:
    from tcia_utils import nbia
except ImportError:
    nbia = None


dataset_keys = Literal[
    "segrap",
    "hanseg",
    "ms_brain",
    "mets_to_brain",
    "hntsmrg",
    "hcc_tace",
    "adrenal_acc",
    "rider_lung",
    "colorectal",
    "lnq",
    "pengwin",
]


DATASET_URLS: dict[dataset_keys, dict] = {
    # ------------------------------- Mendeley Data ------------------------------ #
    "ms_brain": {"source": "mendeley", "dataset_id": "8bctsm8jz7", "size": 0.7},
    # https://data.mendeley.com/datasets/8bctsm8jz7/1
    # ----------------------------------- TCIA ----------------------------------- #
    "hcc_tace": {"source": "tcia", "collection": "hcc-tace-seg", "size": 28.57},
    "adrenal_acc": {"source": "tcia", "collection": "adrenal-acc-ki67-seg", "size": 9.89},
    "rider_lung": {"source": "tcia", "collection": "rider-lungct-seg", "size": 8.53},  #
    "colorectal": {"source": "tcia", "collection": "colorectal-liver-metastases", "size": 10.91},
    "lnq": {"source": "tcia", "collection": "mediastinal-lymph-node-seg", "size": 35.3},
    "mets_to_brain": {"source": "tcia", "collection": "pretreat-metstobrain-masks", "size": 1.7},  # 1.7 GB
    # ---------------------------------- Zenodo ---------------------------------- #
    "hanseg": {"source": "zenodo", "zenodo_id": 7442914, "size": 4.9},
    "hntsmrg": {"source": "zenodo", "zenodo_id": 11199559, "size": 15},
    "pengwin": {"source": "zenodo", "zenodo_id": 10927452, "size": 7.5},
    # ---------------------------------- GDRIVE ---------------------------------- #
    "segrap": {
        "source": "gdrive",
        "url": "https://drive.google.com/drive/folders/115mzmNlZRIewnSR2QFDwW_-RkNM0LC9D",
        "size": 20,
        "pwd": "segrap2023@uestc",
    },
}


def download_from_tcia(collection_name: str, download_dir: Path) -> None:
    """
    Download from TCIA using the NBIA API.
    Unfortunately this can be very unresponsive since TCIA Servers / APIs are bad.
    """

    os.makedirs(download_dir, exist_ok=True)
    data = nbia.getSeries(collection=collection_name)
    nbia.downloadSeries(data, path=str(download_dir), csv_filename=f"{download_dir}/metadata")


def download_from_zenodo(zenodo_id: int, download_dir: Path) -> None:
    zenodo_url = f"https://zenodo.org/api/records/{zenodo_id}"
    download_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(zenodo_url)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        record_data = response.json()

        # Extract the list of files from the metadata
        files = record_data.get("files", [])

        # Return the list of file names and their download links
        file_list = [(file["key"], file["links"]["self"]) for file in files]

        # Open the output file and write the response content in chunks
        for filename, link in file_list:
            if (download_dir / filename).exists():
                continue
            with requests.get(link, stream=True) as r:
                r.raise_for_status()
                with open(download_dir / filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
    else:
        print(f"Failed to download. Status code: {response.status_code}")
    return download_dir


def download_from_gdrive(gdrive_url: str, download_dir: Path) -> None:

    gdrive_link = "https://drive.google.com/drive/folders/115mzmNlZRIewnSR2QFDwW_-RkNM0LC9D?usp=sharing"

    logger.info(f"Downloading dataset from {gdrive_link} to a {download_dir}.")

    download_dir.mkdir(parents=True, exist_ok=True)

    # gauth = GoogleAuth()
    # gauth.LocalWebserverAuth()
    # drive = GoogleDrive(gauth)
    # filelist = drive.ListFile({"q": f"'{folder}' in parents and trashed=false"}).GetList()

    file_list = gdown.download_folder(gdrive_url, str(download_dir), quiet=False, skip_download=True)

    # logger.info(f"Downloading SegRap from {gdrive_link}")
    for name, idx in file_list.items():
        logger.info(f"Downloading {name}")
        if (download_dir / name).exists():
            logger.info(f"File {name} already exists. Skipping Download")
            continue
        else:
            # download_from_gdrive(idx, str(hanseg_temp_dir / name))
            gdown.download(f"https://drive.google.com/uc?id={idx}", str(download_dir / name), quiet=False)

    return


def get_auth_token() -> str:
    """
    Get the authentication token for Mendeley API.
    """
    # Get the authentication token from the environment variable
    token_url = "https://auth.data.mendeley.com/oauth2/authorize"
    client_id = "tassilo.wald@gmail.com"  # input("Enter your Mendeley client ID: ")
    client_secret = "CqTpyBx3e8ePO1pdRnkci02EP8YWnW"  # input("Enter your Mendeley client secret: ")

    payload = {"client_id": client_id, "client_secret": client_secret, "grant_type": "client_credentials"}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.get(token_url, data=payload, headers=headers)

    if response.status_code == 200:
        token_data = response.json()
        return token_data["access_token"]
    else:
        print(f"Error: Unable to obtain access token. Status code: {response.status_code}")
        print(response.text)
        return None


def download_from_mendeley(dataset_id: str, download_dir: Path) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    # Download the dataset from Mendeley
    # The dataset in the repo is public and can be downloaded without any authentication

    logger.warning(
        f"Mendeley Download is currently not working. Please download the dataset manually and deposit it in {download_dir}"
    )
    # dataset_url = f"https://api.data.mendeley.com/datasets/{dataset_id}"
    # access_token = get_auth_token()
    # if access_token is None:
    #     headers = {"Authorization": f"Bearer {access_token}"}
    # # Download the dataset from Mendeley
    # response = requests.get(dataset_url, stream=True, headers=headers)
    # # Check if the request was successful (status code 200)
    # if response.status_code == 200:
    #     # Open the output file and write the response content in chunks
    #     with open(download_dir, "wb") as output_file:
    #         print(f"Downloading {download_dir}...")
    #         for chunk in response.iter_content(chunk_size=8192):
    #             if chunk:  # Filter out keep-alive chunks
    #                 output_file.write(chunk)
    #         print(f"Download complete: {download_dir}")
    # else:
    #     print(f"Failed to download {download_dir}. Status code: {response.status_code}")

    return download_dir


def copy_files_of_modality(modality: int, dataset_root_dir: Path, dataset_target_dir: Path) -> None:
    """
    Copy files of a specific modality from source to target directory.
    """

    for dir_name in ["imagesTr", "imagesTs"]:
        images_dir = dataset_root_dir / dir_name
        if not images_dir.exists():
            logger.warning("Directory {images_dir} does not exist. Skipping...")
            continue

        target_images_dir = dataset_target_dir / dir_name
        target_images_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(list(images_dir.iterdir()), desc=f"Copying {dir_name}..."):
            if img_path.name.endswith(f"_{modality:04d}.nii.gz"):
                new_name = img_path.name.replace(f"_{modality:04d}.nii.gz", "_0000.nii.gz")
                new_img_path = target_images_dir / new_name
                shutil.copy(img_path, new_img_path)

    # Move the dataset json file
    with open(dataset_root_dir / "dataset.json", "r") as f:
        dataset_info = json.load(f)
    dataset_info["channel_names"] = {k: 0 for k, v in dataset_info["channel_names"].items() if v == modality}
    dataset_info["description"] = f"Derivative of {dataset_root_dir.name} dataset with only one modality."
    with open(dataset_target_dir / "dataset.json", "w") as f:
        json.dump(dataset_info, f, indent=4)

    return


def copy_images(dataset_root_dir: Path, dataset_target_dir: Path) -> None:
    """Just copy over the images"""

    for dir_name in ["imagesTr", "imagesTs"]:
        images_dir = dataset_root_dir / dir_name
        if not images_dir.exists():
            logger.warning("Directory {images_dir} does not exist. Skipping...")
            continue

        target_images_dir = dataset_target_dir / dir_name
        target_images_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(list(images_dir.iterdir()), desc=f"Copying {dir_name}..."):
            new_img_path = target_images_dir / img_path.name
            shutil.copy(img_path, new_img_path)

    shutil.copy(dataset_root_dir / "dataset.json", dataset_target_dir / "dataset.json")
    return


def copy_labels_of_modality_and_transform_to_instance(
    dataset_root_dir: Path,
    dataset_target_dir: Path,
    semantic_class_of_interest: int,
    dataset_json_description: str | None = None,
) -> None:
    """
    Create instances of a specific semantic class from the labels of the dataset and move them to the target directory.

    :param dataset_root_dir: Path to the root directory of the dataset.
    :param dataset_target_dir: Path to the target directory of the dataset.
    :param semantic_class_of_interest: The semantic class of interest.
    """
    for lbl_dir in ["labelsTr", "labelsTs"]:
        labels_dir = dataset_root_dir / lbl_dir
        if not labels_dir.exists():
            continue
        assert labels_dir.exists(), f"Labels directory {labels_dir} does not exist."
        target_labels_dir = dataset_target_dir / lbl_dir
        for label_path in tqdm(list(labels_dir.iterdir()), desc=f"Copying {lbl_dir}..."):
            create_instances_from_img(
                label_path,
                output_dir=target_labels_dir,
                semantic_class_of_interest=semantic_class_of_interest,
                dilation_kernel="ball",
                dilation_radius=0,
                connectivity=3,
            )

    with open(dataset_root_dir / "dataset.json", "r") as f:
        old_dataset_info = json.load(f)

    if (target_labels_dir.parent / "dataset.json").exists():
        with open(target_labels_dir.parent / "dataset.json", "r") as f:
            new_dataset_info = json.load(f)
        new_dataset_info["labels"] = {
            k: 1 for k, v in old_dataset_info["labels"].items() if v == semantic_class_of_interest
        }
        if dataset_json_description is not None:
            new_dataset_info["description"] = dataset_json_description
        else:
            new_dataset_info["description"] = f"Instances of {dataset_root_dir.name} dataset."
        with open(target_labels_dir.parent / "dataset.json", "w") as f:
            json.dump(new_dataset_info, f, indent=4)
    else:
        old_dataset_info["labels"] = {
            k: 1 for k, v in old_dataset_info["labels"].items() if v == semantic_class_of_interest
        }
        if dataset_json_description is not None:
            old_dataset_info["description"] = dataset_json_description
        else:
            old_dataset_info["description"] = f"Instances of {dataset_root_dir.name} dataset."
        with open(target_labels_dir.parent / "dataset.json", "w") as f:
            json.dump(old_dataset_info, f, indent=4)


def create_instances_from_img(
    path_to_file: Path,
    output_dir: Path,
    semantic_class_of_interest: int,
    dilation_radius: int = 0,
    dilation_kernel: str = "ball",
    connectivity: int = 3,
) -> None:
    """
    Read an image and create an instance dataset version of it.
    Due to the image being a nifti, no overwrite errors should occur.

    To create instances, the image is binarized for each semantic class and then connected components are calcualted.
    Parameters for that can be provided, such as dilation radius, dilation kernel and connectivity.

    """
    cc_kwargs = {
        "dilation_radius": dilation_radius,
        "dilation_kernel": dilation_kernel,
        "connectivity": connectivity,
    }
    innrrd: InstanceNrrd = InstanceNrrd.from_semantic_img(path_to_file, do_cc=True, cc_kwargs=cc_kwargs)

    ext = "." + ".".join(path_to_file.name.split(".")[1:])
    filename = path_to_file.name.split(".")[0]

    header = innrrd.get_vanilla_header()
    sem_bin_maps = innrrd.get_semantic_instance_maps()
    if semantic_class_of_interest not in sem_bin_maps:
        out_map = np.zeros_like(sem_bin_maps.values()[0][0])
    else:
        instance_bin_maps = sem_bin_maps[semantic_class_of_interest]
        out_map = np.zeros_like(instance_bin_maps[0])
        for i, instance_map in enumerate(instance_bin_maps):
            out_map += instance_map * (i + 1)
    tempdir = tempfile.TemporaryDirectory()
    nrrd.write(tempdir.name + "/tmp.nrrd", out_map, header)
    tmp_img = sitk.ReadImage(tempdir.name + "/tmp.nrrd")
    output_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(tmp_img, output_dir / (filename + ext))
