from pathlib import Path

import torch

from intrab.model.inferer import Inferer
from intrab.prompts.prompt_hparams import PromptConfig
from intrab.model.model_utils import inferer_registry, checkpoint_registry, model_registry

from intrab.experiments import run_experiments_organ, run_experiments_instances
from intrab.utils.io import get_labels_from_dataset_json, get_dataset_path_by_id, get_img_gts, read_yaml_config
from intrab.utils.paths import get_results_path
from argparse import ArgumentParser
from loguru import logger

if __name__ == "__main__":
    # Setup
    # warnings.filterwarnings('error')
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to a configuration '.yaml' file to run the experiments with.",
    )
    args = parser.parse_args()

    # Load the configuration file
    config = read_yaml_config(args.config)
    if torch.cuda.is_available():
        device: str = "cuda"
    else:
        device: str = "cpu"

    # ToDo: Include this in the configuration file.
    exp_params = PromptConfig(
        twoD_n_click_random_points=5,
        twoD_n_slice_point_interpolation=5,
        twoD_n_slice_box_interpolation=5,
        twoD_n_seed_points_point_propagation=5,
        twoD_n_points_propagation=5,
        # interactive_dof_bound=60,
        # interactive_perf_bound=0.9,
        # interactive_max_iter=10,
        interactive_max_iter=5,
        twoD_interactive_n_cc = 1,
        twoD_interactive_n_points_per_slice=1,
        threeD_interactive_n_init_points=1,
        threeD_patch_size=(128, 128, 128),  # This argument is only for SAMMed3D - should be different for segvol
    )

    # Potentially move the seeds inside to not create new models each seed, but save that time.
    wanted_prompt_styles = config["prompting"]["prompt_styles"]
    for seed in config["seeds"]:
        model_name: model_registry
        for dataset in config["datasets"]:
            logger.info(f"Loading dataset {dataset["identifier"]:03d}")
            dataset_root: Path = get_dataset_path_by_id(dataset["identifier"])
            logger.info(f"Dataset loaded from {dataset_root}")
            dataset_name: str = dataset_root.name
            excluded_class_ids: list = dataset['excluded_classes']
            # ToDo: Add the excluded class ids here.
            label_dict: dict[str, int] = get_labels_from_dataset_json(dataset_root, excluded_class_ids)
            imgs_gts: list[tuple[str, str]] = get_img_gts(dataset_root)
            logger.info(f"Found {len(imgs_gts)} image-groundtruth pairs")
            for model_name in config["models"]:
                # Get dataset and the corresponding img, groundtruth Pairs
                results_path = Path(get_results_path() / dataset_name / model_name)
                # Load the model
                checkpoint_path: Path = checkpoint_registry[model_name]
                logger.info(f"Instantiating '{model_name}' with checkpoint '{checkpoint_path}'")
                inferer: Inferer = inferer_registry[model_name](checkpoint_path, device)

                if dataset["type"] == "organ":
                    runner = run_experiments_organ
                else:
                    runner = run_experiments_instances
                runner(
                    inferer,
                    imgs_gts,
                    results_path,
                    label_dict,
                    exp_params,
                    wanted_prompt_styles,
                    seed=1,
                    experiment_overwrite=None,
                    debug=config.get("debug", False),
                )
