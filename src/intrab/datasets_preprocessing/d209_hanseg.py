from pathlib import Path
import zipfile
from intrab.utils.paths import get_dataset_path
import requests


ZENODO_HANSEG_ID = 7442914


def extract_hanseg_dataset(hanseg_temp_dir):
    """Unzip the HaN Seg dataset"""
    for file in hanseg_temp_dir.iterdir():
        print(f"Extracting {file.name}")
        with zipfile.ZipFile(hanseg_temp_dir / file.name, "r") as zip_ref:
            if str(hanseg_temp_dir / file.name).endswith(".zip"):
                # PWD provided on https://han-seg2023.grand-challenge.org/
                zip_ref.extractall(str(hanseg_temp_dir), pwd="segrap2023@uestc".encode("utf-8"))


def main():

    pass


if __name__ == "__main__":
    main()
