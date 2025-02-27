from pathlib import Path
from typing import Literal, Type, Union

from loguru import logger
from radioa.prompts.prompt_hparams import PromptConfig
from radioa.prompts.prompter import (
    Alternating10PointsPer2DSlicePrompter,
    Alternating2PointsPer2DSlicePrompter,
    Alternating3PointsPer2DSlicePrompter,
    Alternating5PointsPer2DSlicePrompter,
    CentroidPoint3DVolumePrompter,
    FiveBoxInterpolationPrompter,
    FivePointInterpolationPrompter,
    FivePointsFromCenterCropped3DVolumePrompter,
    OneFGPointsPer2DSlicePrompter,
    OnePointsFromCenterCropped3DVolumePrompter,
    TenBoxInterpolationPrompter,
    TenPointInterpolationPrompter,
    TenPointsFromCenterCropped3DVolumePrompter,
    ThreeBoxInterpolationPrompter,
    ThreePointInterpolationPrompter,
    ThreePoints3DVolumePrompter,
    ThreePointsFromCenterCropped3DVolumePrompter,
    TwoFGPointsPer2DSlicePrompter,
    ThreeFGPointsPer2DSlicePrompter,
    FiveFGPointsPer2DSlicePrompter,
    TenFGPointsPer2DSlicePrompter,
    Box3DVolumePrompter,
    BoxInterpolationPrompter,
    BoxPer2DSlicePrompter,
    BoxPer2dSliceFrom3DBoxPrompter,
    BoxPropagationPrompter,
    CenterPointPrompter,
    NPoints3DVolumePrompter,
    PointInterpolationPrompter,
    PointPropagationPrompter,
    Prompter,
    OnePoints3DVolumePrompter,
    TwoPoints3DVolumePrompter,
    FivePoints3DVolumePrompter,
    TenPoints3DVolumePrompter,
    TwoPointsFromCenterCropped3DVolumePrompter,
    static_prompt_styles,
)

from radioa.prompts.interactive_prompter import (
    BoxInterpolationInteractivePrompterNoPrevPoint,
    OnePointPer2DSliceInteractivePrompterNoPrevPoint,
    OnePointPer2DSliceInteractivePrompterWithPrevPoint,
    PointInterpolationInteractivePrompterNoPrevPoint,
    PointInterpolationInteractivePrompterWithPrevPoint,
    PointPropagationInteractivePrompterNoPrevPoint,
    PointPropagationInteractivePrompterWithPrevPoint,
    threeDCroppedFromCenterAnd2dAlgoInteractivePrompterNoPrevPoint,
    threeDCroppedFromCenterAnd2dAlgoInteractivePrompterWithPrevPoint,
    threeDCroppedFromCenterInteractivePrompterNoPrevPoint,
    threeDCroppedFromCenterInteractivePrompterWithPrevPoint,
    threeDCroppedInteractivePrompterNoPrevPoint,
    threeDCroppedInteractivePrompterWithPrevPoint,
    twoD1PointUnrealisticInteractivePrompterNoPrevPoint,
    interactive_prompt_styles,
    twoD1PointUnrealisticInteractivePrompterWithPrevPoint,
    BoxInterpolationInteractivePrompterWithPrevBox
)

from radioa.utils.paths import get_model_path
from radioa.model.inferer import Inferer
from radioa.model.SAM import SAMInferer
from radioa.model.SAMMed2D import SAMMed2DInferer
from radioa.model.MedSAM import MedSAMInferer
from radioa.model.SAMMed3D import SAMMed3DInferer
from radioa.model.segvol import SegVolInferer
from radioa.model.SAM2 import SAM2Inferer
from radioa.model.ScribblePrompt import ScribblePromptInferer

model_registry = Literal[
    "sam",
    "sam2",
    "sammed2d",
    "sammed3d",
    "sammed3d_turbo",
    "medsam",
    "segvol",
    "scribbleprompter"
]

inferer_registry: dict[model_registry, Type[Inferer]] = {
    "sam": SAMInferer,
    "sammed2d": SAMMed2DInferer,
    "medsam": MedSAMInferer,
    "sammed3d": SAMMed3DInferer,
    "sammed3d_turbo": SAMMed3DInferer,
    "segvol": SegVolInferer,
    "sam2": SAM2Inferer,
    "scribbleprompter": ScribblePromptInferer
}


checkpoint_registry: dict[model_registry, Path] = {
    "sam": get_model_path() / "sam_vit_h_4b8939.pth",
    "medsam": get_model_path() / "medsam_vit_b.pth",
    "sammed2d": get_model_path() / "sam-med2d_b.pth",
    "segvol": get_model_path() / "SegVol_v1.pth",
    "sammed3d": get_model_path() / "sam_med3d.pth",
    "sammed3d_turbo": get_model_path() / "sam_med3d_turbo.pth",
    "sam2": "",
    "scribbleprompter": ""
}


def get_wanted_supported_prompters(
    inferer: Inferer,
    pro_conf: PromptConfig,
    wanted_prompt_styles: Union[list[static_prompt_styles], list[interactive_prompt_styles]],
    seed: int,
) -> list[Prompter]:
    prompters = []
    if inferer.dim == 2:
        if "point" in inferer.supported_prompts:
            if "OneFGPointsPer2DSlicePrompter" in wanted_prompt_styles:
                prompters.append(
                    OneFGPointsPer2DSlicePrompter(
                        inferer,
                        seed=seed,
                    )
                )
            if "TwoFGPointsPer2DSlicePrompter" in wanted_prompt_styles:
                prompters.append(
                    TwoFGPointsPer2DSlicePrompter(
                        inferer,
                        seed=seed,
                    )
                )
            if "ThreeFGPointsPer2DSlicePrompter" in wanted_prompt_styles:
                prompters.append(
                    ThreeFGPointsPer2DSlicePrompter(
                        inferer,
                        seed=seed,
                    )
                )
            if "FiveFGPointsPer2DSlicePrompter" in wanted_prompt_styles:
                prompters.append(
                    FiveFGPointsPer2DSlicePrompter(
                        inferer,
                        seed=seed,
                    )
                )
            if "TenFGPointsPer2DSlicePrompter" in wanted_prompt_styles:
                prompters.append(
                    TenFGPointsPer2DSlicePrompter(
                        inferer,
                        seed=seed,
                    )
                )
            if "CenterPointPrompter" in wanted_prompt_styles:
                prompters.append(CenterPointPrompter(inferer, seed))
            if "ThreePointInterpolationPrompter" in wanted_prompt_styles:
                prompters.append(
                    ThreePointInterpolationPrompter(
                        inferer,
                        seed=seed,
                    )
                )
            if "FivePointInterpolationPrompter" in wanted_prompt_styles:
                prompters.append(
                    FivePointInterpolationPrompter(
                        inferer,
                        seed=seed,
                    )
                )
            if "TenPointInterpolationPrompter" in wanted_prompt_styles:
                prompters.append(
                    TenPointInterpolationPrompter(
                        inferer,
                        seed=seed,
                    )
                )
            if "PointPropagationPrompter" in wanted_prompt_styles:
                prompters.append(
                    PointPropagationPrompter(
                        inferer,
                        n_seed_points_point_propagation=pro_conf.twoD_n_seed_points_point_propagation,
                        n_points_propagation=pro_conf.twoD_n_points_propagation,
                        seed=seed,
                    )
                )
            if "Alternating2PointsPer2DSlicePrompter" in wanted_prompt_styles:
                prompters.append(
                    Alternating2PointsPer2DSlicePrompter(
                        inferer,
                        seed=seed,
                    )
                )
            if "Alternating3PointsPer2DSlicePrompter" in wanted_prompt_styles:
                prompters.append(
                    Alternating3PointsPer2DSlicePrompter(
                        inferer,
                        seed=seed,
                    )
                )
            if "Alternating5PointsPer2DSlicePrompter" in wanted_prompt_styles:
                prompters.append(
                    Alternating5PointsPer2DSlicePrompter(
                        inferer,
                        seed=seed,
                    )
                )
            if "Alternating10PointsPer2DSlicePrompter" in wanted_prompt_styles:
                prompters.append(
                    Alternating10PointsPer2DSlicePrompter(
                        inferer,
                        seed=seed,
                    )
                )
        if "box" in inferer.supported_prompts:
            if "BoxPer2DSlicePrompter" in wanted_prompt_styles:
                prompters.append(
                    BoxPer2DSlicePrompter(
                        inferer,
                        seed=seed,
                    )
                )
            if "BoxPer2dSliceFrom3DBoxPrompter" in wanted_prompt_styles:
                prompters.append(BoxPer2dSliceFrom3DBoxPrompter(inferer, seed))
            if "ThreeBoxInterpolationPrompter" in wanted_prompt_styles:
                prompters.append(
                    ThreeBoxInterpolationPrompter(
                        inferer,
                        seed=seed,
                    )
                )
            if "FivePointInterpolationPrompter" in wanted_prompt_styles:
                prompters.append(
                    FiveBoxInterpolationPrompter(
                        inferer,
                        seed=seed,
                    )
                )
            if "TenBoxInterpolationPrompter" in wanted_prompt_styles:
                prompters.append(
                    TenBoxInterpolationPrompter(
                        inferer,
                        seed=seed,
                    )
                )
            if "BoxPropagationPrompter" in wanted_prompt_styles:
                prompters.append(BoxPropagationPrompter(inferer, seed))

        if "point" in inferer.supported_prompts and "mask" in inferer.supported_prompts:
            if "OnePointPer2DSliceInteractivePrompterNoPrevPoint" in wanted_prompt_styles:
                prompters.append(
                    OnePointPer2DSliceInteractivePrompterNoPrevPoint(
                        inferer,
                        seed,
                        n_ccs_positive_interaction=pro_conf.twoD_interactive_n_cc,
                        dof_bound=pro_conf.interactive_dof_bound,
                        perf_bound=pro_conf.interactive_perf_bound,
                        max_iter=pro_conf.interactive_max_iter,
                        n_init_points_per_slice=pro_conf.twoD_interactive_n_points_per_slice,
                    )
                )
            if "OnePointPer2DSliceInteractivePrompterWithPrevPoint" in wanted_prompt_styles:
                prompters.append(
                    OnePointPer2DSliceInteractivePrompterWithPrevPoint(
                        inferer,
                        seed,
                        n_ccs_positive_interaction=pro_conf.twoD_interactive_n_cc,
                        dof_bound=pro_conf.interactive_dof_bound,
                        perf_bound=pro_conf.interactive_perf_bound,
                        max_iter=pro_conf.interactive_max_iter,
                        n_init_points_per_slice=pro_conf.twoD_interactive_n_points_per_slice,
                    )
                )
            if "PointInterpolationInteractivePrompterNoPrevPoint" in wanted_prompt_styles:
                prompters.append(
                    PointInterpolationInteractivePrompterNoPrevPoint(
                        inferer,
                        seed,
                        pro_conf.twoD_interactive_n_cc,
                        pro_conf.interactive_dof_bound,
                        pro_conf.interactive_perf_bound,
                        pro_conf.interactive_max_iter,
                        pro_conf.twoD_n_slice_point_interpolation,
                    )
                )
            if "PointInterpolationInteractivePrompterWithPrevPoint" in wanted_prompt_styles:
                prompters.append(
                    PointInterpolationInteractivePrompterWithPrevPoint(
                        inferer,
                        seed,
                        pro_conf.twoD_interactive_n_cc,
                        pro_conf.interactive_dof_bound,
                        pro_conf.interactive_perf_bound,
                        pro_conf.interactive_max_iter,
                        pro_conf.twoD_n_slice_point_interpolation,
                    )
                )
            if "BoxInterpolationInteractivePrompterNoPrevPoint" in wanted_prompt_styles:
                prompters.append(
                    BoxInterpolationInteractivePrompterNoPrevPoint(
                        inferer,
                        seed,
                        pro_conf.twoD_interactive_n_cc,
                        pro_conf.interactive_dof_bound,
                        pro_conf.interactive_perf_bound,
                        pro_conf.interactive_max_iter,
                    )
                )
            if "BoxInterpolationInteractivePrompterWithPrevBox" in wanted_prompt_styles:
                prompters.append(
                    BoxInterpolationInteractivePrompterWithPrevBox(
                        inferer,
                        seed,
                        pro_conf.twoD_interactive_n_cc,
                        pro_conf.interactive_dof_bound,
                        pro_conf.interactive_perf_bound,
                        pro_conf.interactive_max_iter,
                    )
                )
            if "PointPropagationInteractivePrompterNoPrevPoint" in wanted_prompt_styles:
                prompters.append(
                    PointPropagationInteractivePrompterNoPrevPoint(
                        inferer,
                        seed,
                        pro_conf.twoD_interactive_n_cc,
                        pro_conf.interactive_dof_bound,
                        pro_conf.interactive_perf_bound,
                        pro_conf.interactive_max_iter,
                        pro_conf.twoD_n_seed_points_point_propagation,
                        pro_conf.twoD_n_points_propagation,
                    )
                )
            if "PointPropagationInteractivePrompterWithPrevPoint" in wanted_prompt_styles:
                prompters.append(
                    PointPropagationInteractivePrompterWithPrevPoint(
                        inferer,
                        seed,
                        pro_conf.twoD_interactive_n_cc,
                        pro_conf.interactive_dof_bound,
                        pro_conf.interactive_perf_bound,
                        pro_conf.twoD_n_seed_points_point_propagation,
                        pro_conf.twoD_n_points_propagation,
                    )
                )
            if "twoD1PointUnrealisticInteractivePrompterNoPrevPoint" in wanted_prompt_styles:
                prompters.append(
                    twoD1PointUnrealisticInteractivePrompterNoPrevPoint(
                        inferer,
                        seed,
                        dof_bound=pro_conf.interactive_dof_bound,
                        perf_bound=pro_conf.interactive_perf_bound,
                        max_iter=pro_conf.interactive_max_iter,
                        n_init_points_per_slice=pro_conf.twoD_interactive_n_points_per_slice,
                    )
                )
            if "twoD1PointUnrealisticInteractivePrompterWithPrevPoint" in wanted_prompt_styles:
                prompters.append(
                    twoD1PointUnrealisticInteractivePrompterWithPrevPoint(
                        inferer,
                        seed,
                        dof_bound=pro_conf.interactive_dof_bound,
                        perf_bound=pro_conf.interactive_perf_bound,
                        max_iter=pro_conf.interactive_max_iter,
                        n_init_points_per_slice=pro_conf.twoD_interactive_n_points_per_slice,
                    )
                )

    elif inferer.dim == 3:
        if "point" in inferer.supported_prompts:
            if "CentroidPoint3DVolumePrompter" in wanted_prompt_styles:
                prompters.append(CentroidPoint3DVolumePrompter(inferer, seed))
            if "OnePoints3DVolumePrompter" in wanted_prompt_styles:
                prompters.append(OnePoints3DVolumePrompter(inferer, seed))
            if "TwoPoints3DVolumePrompter" in wanted_prompt_styles:
                prompters.append(TwoPoints3DVolumePrompter(inferer, seed))
            if "ThreePoints3DVolumePrompter" in wanted_prompt_styles:
                prompters.append(ThreePoints3DVolumePrompter(inferer, seed))
            if "FivePoints3DVolumePrompter" in wanted_prompt_styles:
                prompters.append(FivePoints3DVolumePrompter(inferer, seed))
            if "TenPoints3DVolumePrompter" in wanted_prompt_styles:
                prompters.append(TenPoints3DVolumePrompter(inferer, seed))

            if "OnePointsFromCenterCropped3DVolumePrompter" in wanted_prompt_styles:
                prompters.append(
                    OnePointsFromCenterCropped3DVolumePrompter(
                        inferer, seed, pro_conf.twoD_n_slice_point_interpolation, pro_conf.threeD_patch_size
                    )
                )
            if "TwoPointsFromCenterCropped3DVolumePrompter" in wanted_prompt_styles:
                prompters.append(
                    TwoPointsFromCenterCropped3DVolumePrompter(
                        inferer, seed, pro_conf.twoD_n_slice_point_interpolation, pro_conf.threeD_patch_size
                    )
                )
            if "ThreePointsFromCenterCropped3DVolumePrompter" in wanted_prompt_styles:
                prompters.append(
                    ThreePointsFromCenterCropped3DVolumePrompter(
                        inferer, seed, pro_conf.twoD_n_slice_point_interpolation, pro_conf.threeD_patch_size
                    )
                )
            if "FivePointsFromCenterCropped3DVolumePrompter" in wanted_prompt_styles:
                prompters.append(
                    FivePointsFromCenterCropped3DVolumePrompter(
                        inferer, seed, pro_conf.twoD_n_slice_point_interpolation, pro_conf.threeD_patch_size
                    )
                )
            if "TenPointsFromCenterCropped3DVolumePrompter" in wanted_prompt_styles:
                prompters.append(
                    TenPointsFromCenterCropped3DVolumePrompter(
                        inferer, seed, pro_conf.twoD_n_slice_point_interpolation, pro_conf.threeD_patch_size
                    )
                )

        if "box" in inferer.supported_prompts:
            if "Box3DVolumePrompter" in wanted_prompt_styles:
                prompters.append(Box3DVolumePrompter(inferer, seed))
        if "point" in inferer.supported_prompts and "mask" in inferer.supported_prompts:
            if "threeDCroppedInteractivePrompterNoPrevPoint" in wanted_prompt_styles:
                prompters.append(
                    threeDCroppedInteractivePrompterNoPrevPoint(
                        inferer,
                        pro_conf.threeD_interactive_n_init_points,
                        seed,
                        pro_conf.interactive_dof_bound,
                        pro_conf.interactive_perf_bound,
                        pro_conf.interactive_max_iter,
                        pro_conf.threeD_patch_size,
                    )
                )
            if "threeDCroppedInteractivePrompterWithPrevPoint" in wanted_prompt_styles:
                prompters.append(
                    threeDCroppedInteractivePrompterWithPrevPoint(
                        inferer,
                        pro_conf.threeD_interactive_n_init_points,
                        seed,
                        pro_conf.interactive_dof_bound,
                        pro_conf.interactive_perf_bound,
                        pro_conf.interactive_max_iter,
                        pro_conf.threeD_patch_size,
                    )
                )
            if "threeDCroppedFromCenterInteractivePrompterNoPrevPoint" in wanted_prompt_styles:
                prompters.append(
                    threeDCroppedFromCenterInteractivePrompterNoPrevPoint(
                        inferer,
                        seed,
                        pro_conf.threeD_interactive_n_init_points,
                        pro_conf.interactive_dof_bound,
                        pro_conf.interactive_perf_bound,
                        pro_conf.interactive_max_iter,
                        pro_conf.threeD_patch_size,
                        pro_conf.twoD_n_slice_point_interpolation,
                    )
                )
            if "threeDCroppedFromCenterInteractivePrompterWithPrevPoint" in wanted_prompt_styles:
                prompters.append(
                    threeDCroppedFromCenterInteractivePrompterWithPrevPoint(
                        inferer,
                        seed,
                        pro_conf.threeD_interactive_n_init_points,
                        pro_conf.interactive_dof_bound,
                        pro_conf.interactive_perf_bound,
                        pro_conf.interactive_max_iter,
                        pro_conf.threeD_patch_size,
                        pro_conf.twoD_n_slice_point_interpolation,
                    )
                )
            if "threeDCroppedFromCenterAnd2dAlgoInteractivePrompterNoPrevPoint" in wanted_prompt_styles:
                prompters.append(
                    threeDCroppedFromCenterAnd2dAlgoInteractivePrompterNoPrevPoint(
                        inferer,
                        seed,
                        pro_conf.threeD_interactive_n_init_points,
                        pro_conf.interactive_dof_bound,
                        pro_conf.interactive_perf_bound,
                        pro_conf.interactive_max_iter,
                        pro_conf.threeD_patch_size,
                        pro_conf.twoD_n_slice_point_interpolation,
                        pro_conf.threeD_interactive_n_corrective_points,
                        pro_conf.twoD_interactive_n_cc,
                    )
                )
            if "threeDCroppedFromCenterAnd2dAlgoInteractivePrompterWithPrevPoint" in wanted_prompt_styles:
                prompters.append(
                    threeDCroppedFromCenterAnd2dAlgoInteractivePrompterWithPrevPoint(
                        inferer,
                        seed,
                        pro_conf.threeD_interactive_n_init_points,
                        pro_conf.interactive_dof_bound,
                        pro_conf.interactive_perf_bound,
                        pro_conf.interactive_max_iter,
                        pro_conf.threeD_patch_size,
                        pro_conf.twoD_n_slice_point_interpolation,
                        pro_conf.threeD_interactive_n_corrective_points,
                        pro_conf.twoD_interactive_n_cc,
                    )
                )
    else:
        raise ValueError(f"Inferer dimension '{inferer.dim}' not supported. Choose from [2, 3]")

    if len(prompters) == 0:
        logger.warning(f"No prompters selected; only evaluation will be performed.")
    return prompters
