from pathlib import Path
from typing import Literal, Type

from loguru import logger

from intrab.prompts.prompt_hparams import PromptConfig
from intrab.prompts.prompter import (
    OneFGPointsPer2DSlicePrompter,
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
    static_prompt_styles,
)

from intrab.prompts.interactive_prompter import (
    NPointsPer2DSliceInteractivePrompter,
    threeDInteractivePrompterSAMMed3D,
    twoDNPointsUnrealisticInteractivePrompter,
    interactive_prompt_styles,
)

from intrab.utils.paths import get_model_path
from intrab.model.inferer import Inferer
from intrab.model.SAM import SAMInferer
from intrab.model.SAMMed2D import SAMMed2DInferer
from intrab.model.MedSAM import MedSAMInferer
from intrab.model.SAMMed3D import SAMMed3DInferer
from intrab.model.segvol import SegVolInferer


model_registry = Literal["sam", "sammed2d", "sammed3d", "sammed3d_turbo", "medsam", "segvol"]

inferer_registry: dict[model_registry, Type[Inferer]] = {
    "sam": SAMInferer,
    "sammed2d": SAMMed2DInferer,
    "medsam": MedSAMInferer,
    "sammed3d": SAMMed3DInferer,
    "sammed3d_turbo": SAMMed3DInferer,
    "segvol": SegVolInferer,
}


checkpoint_registry: dict[model_registry, Path] = {
    "sam": get_model_path() / "sam_vit_h_4b8939.pth",
    "medsam": get_model_path() / "medsam_vit_b.pth",
    "sammed2d": get_model_path() / "sam-med2d_b.pth",
    "segvol": get_model_path() / "SegVol_v1.pth",
    "sammed3d": get_model_path() / "sam_med3d.pth",
    "sammed3d_turbo": get_model_path() / "sam_med3d_turbo.pth",
}


def get_wanted_supported_prompters(
    inferer: Inferer,
    pro_conf: PromptConfig,
    wanted_prompt_styles: list[static_prompt_styles],
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
            if "PointInterpolationPrompter" in wanted_prompt_styles:
                prompters.append(
                    PointInterpolationPrompter(
                        inferer,
                        n_slice_point_interpolation=pro_conf.twoD_n_slice_point_interpolation,
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
            if "BoxInterpolationPrompter" in wanted_prompt_styles:
                prompters.append(
                    BoxInterpolationPrompter(
                        inferer,
                        seed,
                        n_slice_box_interpolation=pro_conf.twoD_n_slice_box_interpolation,
                    )
                )
            if "BoxPropagationPrompter" in wanted_prompt_styles:
                prompters.append(BoxPropagationPrompter(inferer, seed))

        if "point" in inferer.supported_prompts and "mask" in inferer.supported_prompts:
            if "NPointsPer2DSliceInteractive" in wanted_prompt_styles:
                prompters.append(
                    NPointsPer2DSliceInteractivePrompter(
                        inferer,
                        seed,
                        dof_bound=pro_conf.interactive_dof_bound,
                        perf_bound=pro_conf.interactive_perf_bound,
                        max_iter=pro_conf.interactive_max_iter,
                        n_init_points_per_slice=pro_conf.twoD_interactive_n_points_per_slice,
                    )
                )
            if "twoDNPointsUnrealisticInteractivePrompter" in wanted_prompt_styles:
                prompters.append(
                    twoDNPointsUnrealisticInteractivePrompter(
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
        if "box" in inferer.supported_prompts:
            if "Box3DVolumePrompter" in wanted_prompt_styles:
                prompters.append(Box3DVolumePrompter(inferer, seed))
        if "point" in inferer.supported_prompts and "mask" in inferer.supported_prompts:
            if "threeDInteractivePrompter" in wanted_prompt_styles:
                prompters.append(
                    threeDInteractivePrompterSAMMed3D(
                        inferer,
                        pro_conf.threeD_interactive_n_init_points,
                        seed,
                        pro_conf.interactive_dof_bound,
                        pro_conf.interactive_perf_bound,
                        pro_conf.interactive_max_iter,
                        pro_conf.threeD_patch_size,
                    )
                )
    else:
        raise ValueError(f"Inferer dimension '{inferer.dim}' not supported. Choose from [2, 3]")

    if len(prompters) == 0:
        logger.warning(f"No prompters selected; only evaluation will be performed.")
    return prompters
