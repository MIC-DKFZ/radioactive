from dataclasses import dataclass


@dataclass
class PromptConfig:
    twoD_n_click_random_points: int = 5
    twoD_n_slice_point_interpolation: int = 5
    twoD_n_slice_box_interpolation: int = 5
    twoD_n_seed_points_point_propagation: int = 5
    twoD_n_points_propagation: int = 5
    threeD_n_click_random_points: int = 5
    interactive_dof_bound: int = None
    interactive_perf_bound: float = None
    interactive_max_iter: int = None
    twoD_interactive_n_cc: int = 1
    twoD_interactive_n_points_per_slice: int = 1
    threeD_interactive_n_init_points: int = 1
    threeD_patch_size: tuple[int, int, int] = None
    threeD_interactive_n_corrective_points: int = 1
    