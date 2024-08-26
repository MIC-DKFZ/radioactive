import os
from pathlib import Path
from loguru import logger

def get_dataset_path(config: dict) -> Path:
    """
    Path to directory containing all (downloaded/trained) models used to extract representations, datasets, and results.
    Can be overridden by setting the environment variable 'REP_SIM'.
    Will contain subdirectories of `nlp`, `graph`, `vision`, and `results`.
    """
    if config is None or config["INTRAB_DATA_PATH"] is None:
        logger.info("No 'INTRAB_DATA_PATH' in config)")
        
        return Path(__file__).parent.parent.parent / "data"
    try:
        EXPERIMENTS_ROOT_PATH = os.environ["REP_SIM"]  # To be renamed to ones liking
        return Path(EXPERIMENTS_ROOT_PATH)
    except KeyError:
        logger.warning("No 'DATA_RESULTS_FOLDER' Env variable -- Defaulting to '<project_root>/experiments' .")
        exp_pth = Path(__file__).parent.parent.parent / "experiments"
        exp_pth.mkdir(exist_ok=True)
        return exp_pth
