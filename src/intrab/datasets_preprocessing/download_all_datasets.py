from typing import get_args
from src.intrab.datasets_preprocessing.utils import (
    DATASET_URLS,
    dataset_keys,
    download_from_tcia,
    download_from_zenodo,
    download_from_gdrive,
    download_from_mendeley,
)
from argparse import ArgumentParser
from loguru import logger


def download_datasets(datasets_to_download: dict[dataset_keys, dict]):

    for dataset_key, infos in datasets_to_download.items():
        if infos["source"] == "zenodo":
            download_from_zenodo(infos["id"], infos["path"])
        elif infos["source"] == "tcia":
            download_from_tcia(infos["collection"], infos["path"])
        elif infos["source"] == "gdrive":
            download_from_gdrive(infos["id"], infos["path"])
        elif infos["source"] == "mendeley":
            download_from_mendeley(infos["id"], infos["path"])
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--datasets", nargs="+", help="Datasets to download", choices=get_args(dataset_keys))
    args = parser.parse_args()

    datasets_to_download = {dataset_key: DATASET_URLS[dataset_key] for dataset_key in args.datasets}

    download_datasets(datasets_to_download)
