import os
from pathlib import Path
from loguru import logger


def get_dataset_path() -> Path:
    """
    Path to directory containing all (downloaded/trained) models used to extract representations, datasets, and results.
    Can be overridden by setting the environment variable 'REP_SIM'.
    Will contain subdirectories of `nlp`, `graph`, `vision`, and `results`.
    """
    try:
        EXPERIMENTS_ROOT_PATH = os.environ["RADIOA_DATA_PATH"]  # To be renamed to ones liking
        return Path(EXPERIMENTS_ROOT_PATH)
    except KeyError:
        logger.error("No 'RADIOA_DATA_PATH' Env variable set.")
        raise KeyError("No 'RADIOA_DATA_PATH' Env variable set.")


def get_model_path() -> Path:
    """
    Path to directory containing all (downloaded/trained) models used to extract representations, datasets, and results.
    Can be overridden by setting the environment variable 'REP_SIM'.
    Will contain subdirectories of `nlp`, `graph`, `vision`, and `results`.
    """
    try:
        EXPERIMENTS_ROOT_PATH = os.environ["RADIOA_MODEL_PATH"]  # To be renamed to ones liking
        return Path(EXPERIMENTS_ROOT_PATH)
    except KeyError:
        logger.error("No 'RADIOA_MODEL_PATH' Env variable set.")
        raise KeyError("No 'RADIOA_MODEL_PATH' Env variable set.")


def get_results_path() -> Path:
    """
    Path to directory containing all (downloaded/trained) models used to extract representations, datasets, and results.
    Can be overridden by setting the environment variable 'REP_SIM'.
    Will contain subdirectories of `nlp`, `graph`, `vision`, and `results`.
    """
    try:
        EXPERIMENTS_ROOT_PATH = os.environ["RADIOA_RESULTS_PATH"]  # To be renamed to ones liking
        return Path(EXPERIMENTS_ROOT_PATH)
    except KeyError:
        logger.error("No 'RADIOA_RESULTS_PATH' Env variable set.")
        raise KeyError("No 'RADIOA_RESULTS_PATH' Env variable set.")


def get_MITK_path() -> Path:
    """
    Path to directory containing all (downloaded/trained) models used to extract representations, datasets, and results.
    Can be overridden by setting the environment variable 'REP_SIM'.
    Will contain subdirectories of `nlp`, `graph`, `vision`, and `results`.
    """
    if "RADIOA_MITK_PATH" in os.environ:
        return Path(os.environ["RADIOA_MITK_PATH"])
    else:
        mitk_path = get_dataset_path().parent / "MITK"
        mitk_path.mkdir(exist_ok=True, parents=True)
        return mitk_path
