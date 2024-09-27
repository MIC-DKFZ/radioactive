from dataclasses import dataclass
from nibabel import Nifti1Image
from toinstance import InstanceNrrd
import numpy as np

from intrab.prompts.prompt import PromptStep


@dataclass
class PromptResult:
    """Result data class."""

    predicted_image: Nifti1Image | InstanceNrrd
    logits: np.ndarray
    prompt_step: PromptStep
    perf: float
    n_step: int
    dof: int
