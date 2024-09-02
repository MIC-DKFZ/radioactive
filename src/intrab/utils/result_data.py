from dataclasses import dataclass
from nibabel import Nifti1Image
import numpy as np

from src.intrab.prompts.prompt import PromptStep


@dataclass
class PromptResult:
    """Result data class."""

    predicted_image: Nifti1Image
    logits: np.ndarray
    prompt_step: PromptStep
    perf: float
    n_step: int
    dof: int
