from argparse import ArgumentParser
from typing import Callable, get_args

from loguru import logger
from intrab.datasets_preprocessing.utils import dataset_keys
from intrab.utils.paths import get_dataset_path
from intrab.datasets_preprocessing.d201_ms_flair import preprocess as d201_preprocess_msflair

# from intrab.datasets_preprocessing.d204_brainmetshare import preprocess as d204_preprocess_brainmetshare
from intrab.datasets_preprocessing.d207_yale_brainmets import preprocess as d207_preprocess_yale_mets_to_brain
from intrab.datasets_preprocessing.d209_hanseg import preprocess as d209_preprocess_hanseg
from intrab.datasets_preprocessing.d501_hntsmrg import preprocess as d501_preprocess_hntsmrg
from intrab.datasets_preprocessing.d600_pengwin import preprocess as d600_preprocess_pengwin
from intrab.datasets_preprocessing.d651_segrap import preprocess as d651_preprocess_segrap
from intrab.datasets_preprocessing.d911_lnq import preprocess as d911_preprocess_lnq
from intrab.datasets_preprocessing.d912_colorectal_livermets import preprocess as d912_preprocess_colorectal_livermets
from intrab.datasets_preprocessing.d913_acc_ki67 import preprocess as d913_preprocess_acc_ki67
from intrab.datasets_preprocessing.d920_d921_hcc_tace_liver import preprocess as d920_d921_preprocess_hcc_tace_liver
from intrab.datasets_preprocessing.d930_rider_lungct import preprocess as d930_preprocess_rider_lungct


preprocessed_dataset_keys: dict[dataset_keys, Callable[str, [None]]] = {
    "ms_flair": d201_preprocess_msflair,  # MR Lesion
    "mets_to_brain": d207_preprocess_yale_mets_to_brain,  # MR Lesion
    "hanseg": d209_preprocess_hanseg,  # MR Organ
    "hntsmrg": d501_preprocess_hntsmrg,  # MR Lesion
    "pengwin": d600_preprocess_pengwin,  # CT Bone
    "segrap": d651_preprocess_segrap,  # CT Organ
    "lnq": d911_preprocess_lnq,  # CT Lesion
    "colorectal": d912_preprocess_colorectal_livermets,  # CT Lesion
    "adrenal_acc": d913_preprocess_acc_ki67,  # CT Lesion
    "hcc_tace": d920_d921_preprocess_hcc_tace_liver,  # CT Lesion
    "rider_lung": d930_preprocess_rider_lungct,  # CT Lesion
}


def preprocess_datasets(ds_keys: list[dataset_keys]):
    """
    Starts the preprocssing of datasets previously downloaded.
    """
    download_path = get_dataset_path().parent / "raw_dataset_downloads"
    for ds in ds_keys:
        logger.info(f"Starting preprocessing of {ds} dataset.")
        preprocessed_dataset_keys[ds](download_path / ds)
        logger.info(f"Finished preprocessing of {ds} dataset.")


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

    datasets_to_download = list(args.datasets)

    preprocess_datasets(datasets_to_download)
