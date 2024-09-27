# Experiments content
import os
from pathlib import Path

from loguru import logger
from intrab.datasets_preprocessing.conversion_utils import nib_to_nrrd
from intrab.model.inferer import Inferer
from intrab.model.model_utils import get_wanted_supported_prompters
from intrab.prompts.prompt_hparams import PromptConfig
from intrab.prompts.prompter import static_prompt_styles
from toinstance import InstanceNrrd
from intrab.prompts.prompter import Prompter


from tqdm import tqdm

from intrab.utils.io import (
    binarize_gt,
    create_instance_gt,
    save_instance_results,
    save_nib_instance_gt_as_nrrd,
    verify_results_dir_exist,
)
from intrab.utils.io import verify_results_dir_exist
from nneval.evaluate_instance import instance_evaluation
from nneval.evaluate_semantic import semantic_evaluation

from intrab.utils.result_data import PromptResult


def run_experiments_organ(
    inferer: Inferer,
    imgs_gts: list[tuple[str, str]],
    results_dir: Path,
    label_dict: dict[str, int],
    pro_conf: PromptConfig,
    wanted_prompt_styles: list[static_prompt_styles],
    seed,
    experiment_overwrite=None,
    results_overwrite: bool = False,
    debug: bool = False,
    only_calc: bool = False,
    only_eval: bool = False,
):
    targets: dict = {k.replace("/", "_"): v for k, v in label_dict.items()}
    prompters: list[Prompter] = get_wanted_supported_prompters(inferer, pro_conf, wanted_prompt_styles, seed)
    verify_results_dir_exist(targets=targets, results_dir=results_dir, prompters=prompters)

    if debug:
        logger.warning("Debug mode activated. Only running on the first three images.")
        imgs_gts = imgs_gts[:3]

    logger.warning(
        "Coordinate systems should be checked and verified for correctness. \n"
        + "Right now this is assumed to be correct"
    )

    is_on_cluster = "LSF_JOBID" in os.environ

    if not only_eval:
        # Loop through all image and label pairs
        target_names: set[str] = set()
        for img_path, gt_path in tqdm(imgs_gts, desc="looping through files", leave=False, disable=is_on_cluster):
            # Loop through each organ label except the background
            for target, target_label in tqdm(
                targets.items(), desc="Predicting targets...", leave=False, disable=True
            ):
                target_name: str = f"{target_label:03d}" + "__" + target
                target_names.add(target_name)

                # ---------------- Get binary gt in original coordinate system --------------- #
                base_name = os.path.basename(gt_path)
                base_name = base_name.replace(".nrrd", ".nii.gz") if base_name.endswith(".nrrd") else base_name
                bin_gt_filepath = results_dir.parent / "binarised_gts" / target_name / base_name
                binary_gt_orig_coords = binarize_gt(gt_path, target_label)
                if not bin_gt_filepath.exists():
                    bin_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
                    binary_gt_orig_coords.to_filename(bin_gt_filepath)

                for prompter in tqdm(
                    prompters,
                    desc="Prompting with various prompters ...",
                    leave=False,
                    disable=True,
                ):
                    filepath = results_dir / prompter.name / target_name / base_name
                    if filepath.exists() and not results_overwrite:
                        # logger.debug(f"Skipping {gt_path} as it has already been processed.")
                        continue
                    prompter.set_groundtruth(binary_gt_orig_coords)

                    # Handle non-interactive experiments
                    if prompter.is_static:
                        prediction_result: PromptResult
                        prediction_result = prompter.predict_image(image_path=img_path)
                        # Save the prediction
                        prediction_result.predicted_image.to_filename(filepath)
                    # Handle interactive experiments
                    else:
                        prediction_results: list[PromptResult]
                        prediction_results = prompter.predict_image(image_path=img_path)
                        base_path = filepath.parent

                        for cnt, pred in enumerate(prediction_results):
                            target_path = base_path / f"iter_{cnt}"
                            target_path.mkdir(exist_ok=True)
                            pred.predicted_image.to_filename(target_path / base_name)

    if not only_calc:
        # We always run the semantic eval on the created folders directly.
        for target_name in target_names:
            for prompter in prompters:
                if prompter.is_static:
                    with logger.catch(level="WARNING"):
                        semantic_evaluation(
                            semantic_gt_path=results_dir.parent / "binarised_gts" / target_name,
                            semantic_pd_path=results_dir / prompter.name / target_name,
                            output_path=results_dir / prompter.name,
                            classes_of_interest=(1,),
                            output_name=target_name,
                        )
                else:
                    for i in range(prompter.num_iterations):
                        pred_path = results_dir / prompter.name / target_name / f"iter_{i}"
                        if pred_path.exists():
                            with logger.catch(level="WARNING"):
                                semantic_evaluation(
                                    semantic_gt_path=results_dir.parent / "binarised_gts" / target_name,
                                    semantic_pd_path=pred_path,
                                    output_path=results_dir / prompter.name / f"iter_{i}_eval",
                                    classes_of_interest=(1,),
                                    output_name=target_name,
                                )


def run_experiments_instances(
    inferer: Inferer,
    imgs_gts: list[tuple[str, str]],
    results_dir: Path,
    label_dict: dict[str, int],
    pro_conf: PromptConfig,
    wanted_prompt_styles: list[static_prompt_styles],
    seed,
    experiment_overwrite=None,
    results_overwrite: bool = False,
    debug: bool = False,
    only_eval: bool = False,
    only_calc: bool = False,
):
    targets: dict = {k.replace("/", "_"): v for k, v in label_dict.items() if k.lower() != "background"}
    assert len(targets) == 1, "Instance experiments only support two classes -- background and the instance class"
    prompters: list[Prompter] = get_wanted_supported_prompters(inferer, pro_conf, wanted_prompt_styles, seed)
    verify_results_dir_exist(targets=targets, results_dir=results_dir, prompters=prompters)

    if debug:
        logger.warning("Debug mode activated. Only running on the first three images.")
        imgs_gts = imgs_gts[:3]

    logger.warning(
        "Coordinate systems should be checked and verified for correctness. \n"
        + "Right now this is assumed to be correct"
    )
    gt_output_path = results_dir.parent / "instance_gts"  # GTs are the same for all Models.
    pred_output_path = results_dir
    results_dir.mkdir(exist_ok=True, parents=True)
    gt_output_path.mkdir(exist_ok=True)
    is_on_cluster = "LSF_JOBID" in os.environ

    target_label = list(targets.values())[0]
    target = list(targets.keys())[0]
    target_name = f"{target_label:03d}" + "__" + target

    if not only_eval:
        # Loop through all image and label pairs
        for img_path, gt_path in tqdm(imgs_gts, desc="looping through files ...", leave=False, disable=is_on_cluster):
            # Loop through each organ label except the background
            filename = os.path.basename(gt_path)
            filename_wo_ext = filename.split(".")[0]  # Remove the extension
            instance_filename = f"{filename_wo_ext}.in.nrrd"
            instance_gt_path = gt_output_path / instance_filename

            # ---------------- Get binary gt in original coordinate system --------------- #
            instance_nib, instance_ids = create_instance_gt(gt_path)
            if not instance_gt_path.exists():
                save_nib_instance_gt_as_nrrd(instance_nib, instance_gt_path)

            prompter: Prompter
            for prompter in tqdm(
                prompters,
                desc="Prompting with various prompters ...",
                leave=False,
                disable=True,
            ):
                prompt_pred_path = pred_output_path / prompter.name
                prompt_pred_path.mkdir(exist_ok=True)

                # --------------- Skip prediction if the results already exist. -------------- #
                if prompter.is_static:
                    instance_pd_path = prompt_pred_path / target_name / instance_filename

                    if instance_pd_path.exists() and not results_overwrite:
                        # logger.debug(f"Skipping {gt_path} as it has already been processed.")
                        continue
                else:
                    expected_ins_paths = [
                        prompt_pred_path / f"iter_{i}" / instance_filename for i in range(prompter.num_iterations)
                    ]
                    if all([path.exists() for path in expected_ins_paths]) and not results_overwrite:
                        # logger.debug(f"Skipping {gt_path} as it has already been processed.")
                        continue

                all_prompt_results: list[PromptResult] = []
                for instance_id in instance_ids:
                    binary_gt_orig_coords = binarize_gt(gt_path, instance_id)
                    prompter.set_groundtruth(binary_gt_orig_coords)

                    if prompter.is_static:
                        prediction_result: PromptResult
                        prediction_result = prompter.predict_image(image_path=img_path)
                        # Save the prediction
                        if len(all_prompt_results) == 0:
                            arr, head = nib_to_nrrd(prediction_result.predicted_image)
                            inrrd = InstanceNrrd.from_instance_map(arr, head, 1)
                            prediction_result.predicted_image = inrrd
                            all_prompt_results.append(prediction_result)
                        else:
                            # Currently we don't do any fancy saving of the prompt,
                            #  so we can also just drop it for now.
                            arr, _ = nib_to_nrrd(prediction_result.predicted_image)
                            all_prompt_results[0].predicted_image.add_maps({1: [arr]})

                        # Handle interactive experiments
                    else:
                        prediction_results: list[PromptResult] = prompter.predict_image(image_path=img_path)
                        if len(all_prompt_results) == 0:
                            for pred in prediction_results:
                                arr, head = nib_to_nrrd(pred.predicted_image)
                                inrrd = InstanceNrrd.from_instance_map(arr, head, 1)
                                pred.predicted_image = inrrd
                                all_prompt_results.append(pred)
                        else:
                            for cnt, pred in enumerate(prediction_results):
                                arr, _ = nib_to_nrrd(pred.predicted_image)
                                all_prompt_results[cnt].predicted_image.add_maps({1: [arr]})
                    # -------------------- Save static or interactive results -------------------- #
                save_instance_results(all_prompt_results, (prompt_pred_path / target_name), instance_filename)

    if not only_calc:
        # # We always run the semantic eval on the created folders directly.
        for prompter in prompters:
            if prompter.is_static:
                with logger.catch(level="WARNING"):
                    instance_evaluation(
                        instance_gt_path=gt_output_path,
                        instance_pd_path=pred_output_path / prompter.name,
                        output_path=pred_output_path / prompter.name,
                        classes_of_interest=(1,),
                        dice_threshold=1e-9,
                    )
            else:
                for i in range(prompter.num_iterations):
                    pred_path = pred_output_path / prompter.name / f"iter_{i}"
                    if pred_path.exists():
                        with logger.catch(level="WARNING"):
                            instance_evaluation(
                                instance_gt_path=gt_output_path,
                                instance_pd_path=pred_output_path / prompter.name / f"iter_{i}",
                                output_path=pred_output_path / prompter.name / f"iter_{i}",
                                classes_of_interest=(1,),
                                dice_threshold=1e-9,
                            )
