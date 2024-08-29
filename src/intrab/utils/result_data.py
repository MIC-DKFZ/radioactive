from dataclasses import dataclass
from nibabel import Nifti1Image
import numpy as np


@dataclass
class PromptResult:
    """Result data class."""

    predicted_image: Nifti1Image
    logits: np.ndarray
    perf: float
    n_step: int
    dof: int
