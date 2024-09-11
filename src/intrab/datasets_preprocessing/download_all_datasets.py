from pathlib import Path
from typing import get_args
import zipfile
from intrab.utils.paths import get_dataset_path
from intrab.datasets_preprocessing.utils import (
    DATASET_URLS,
    dataset_keys,
    download_from_tcia,
    download_from_zenodo,
    download_from_gdrive,
    download_from_mendeley,
)
from argparse import ArgumentParser
from loguru import logger
import glob


def extract_zips_in_dir(dataset_dict: dict):
    """Unzip the HaN Seg dataset"""
    pwd: str = dataset_dict.get("pwd", None)
    zip_files = glob.glob(str(dataset_dict / "*.zip"))
    for file in zip_files:
        file_path = Path(file)
        print(f"Extracting {file}")
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            # PWD provided on https://han-seg2023.grand-challenge.org/
            if pwd is not None:
                zip_ref.extractall(str(file_path), pwd=pwd.encode("utf-8"))
            else:
                zip_ref.extractall(str(file_path))


def download_datasets(datasets_to_download: dict[dataset_keys, dict]):
    """
    Starts the download of a dataset and saves in `raw_dataset_downloads` folder next to the `dataset` folder.
    :param datasets_to_download: A dictionary with the datasets to download. -- Contains info about the dataset and where to get it from.

    :return: None
    """
    download_path = get_dataset_path().parent / "raw_dataset_downloads"
    download_path.mkdir(exist_ok=True)
    for dataset_key, infos in datasets_to_download.items():
        cur_dataset_path = download_path / dataset_key
        cur_dataset_path.mkdir(exist_ok=True)
        logger.info(f"Starting download of {dataset_key} dataset.")
        if infos["source"] == "zenodo":
            download_from_zenodo(infos["id"], cur_dataset_path)
        elif infos["source"] == "tcia":
            download_from_tcia(infos["collection"], cur_dataset_path)
        elif infos["source"] == "gdrive":
            download_from_gdrive(infos["id"], cur_dataset_path)
        elif infos["source"] == "mendeley":
            download_from_mendeley(infos["id"], cur_dataset_path)
        logger.info("Extracting zips in dataset directory")
        extract_zips_in_dir(cur_dataset_path)
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Datasets to download",
        choices=get_args(dataset_keys),
        default=list(get_args(dataset_keys)),
    )
    args = parser.parse_args()

    datasets_to_download = {dataset_key: DATASET_URLS[dataset_key] for dataset_key in args.datasets}

    download_datasets(datasets_to_download)
