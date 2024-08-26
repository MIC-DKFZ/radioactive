from dataclasses import dataclass


@dataclass
class PromptConfig2D:
    n_click_random_points: int = 5
    n_slice_point_interpolation: int = 5
    n_slice_box_interpolation: int = 5
    n_seed_points_point_propagation: int = 5
    n_points_propagation: int = 5
    dof_bound: int = 60
    perf_bound: float = 0.85


@dataclass
class PromptConfig3D:
    n_click_random_points: int = 5
    dof_bound: int = 24
    perf_bound: float = 0.85
