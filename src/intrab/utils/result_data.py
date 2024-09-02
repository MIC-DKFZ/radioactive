from dataclasses import dataclass
import numpy as np


@dataclass
class PromptMetaResult:
    """Result data class."""

    logits: np.ndarray
    perf: float
    n_step: int
    dof: int
