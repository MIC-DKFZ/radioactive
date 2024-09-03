from dataclasses import dataclass


@dataclass
class PromptConfig:
    twoD_n_click_random_points: int = 5
    twoD_n_slice_point_interpolation: int = 5
    twoD_n_slice_box_interpolation: int = 5
    twoD_n_seed_points_point_propagation: int = 5
    twoD_n_points_propagation: int = 5
    threeD_n_click_random_points: int = 5
    interactive_dof_bound: int = 60
    interactive_perf_bound: float = 0.9
    interactive_max_iter: int = 10
    twoD_interactive_n_points_per_slice: int = 5
    threeD_interactive_n_init_points: int = 5
    threeD_patch_size: tuple[int, int, int] = None
    