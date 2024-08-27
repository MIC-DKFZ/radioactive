from dataclasses import dataclass


@dataclass
class PromptConfig:
    twoD_n_click_random_points: int = 5
    twoD_n_slice_point_interpolation: int = 5
    twoD_n_slice_box_interpolation: int = 5
    twoD_n_seed_points_point_propagation: int = 5
    twoD_n_points_propagation: int = 5
    twoD_dof_bound: int = 60
    twoD_perf_bound: float = 0.85
    threeD_n_click_random_points: int = 5
    threeD_dof_bound: int = 24
    threeD_perf_bound: float = 0.85
