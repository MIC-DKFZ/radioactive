from typing import Literal
import numpy as np
import pytest

from radioa.prompts.prompt_3d import get_bbox3d, get_pos_clicks3D
from radioa.prompts.prompt_utils import (
    box_interpolation,
    get_bbox3d_sliced,
    get_minimal_boxes_row_major,
    get_pos_clicks2D_row_major,
    get_seed_boxes,
    get_seed_point,
    point_interpolation,
)


locations = Literal["bottom", "center", "top"]

"""
Tests verifying that all prompters can handle a single receiving a single point.
Guarantees that e.g. multi click logic does not break when fewer foreground than clicks exist.

"""


def get_single_point_gt(x: locations, y: locations, z: locations) -> np.ndarray:
    gt_arr = np.zeros(shape=(32, 32, 32))
    if x == "bottom":
        x_loc = 0
    elif x == "center":
        x_loc = 16
    elif x == "top":
        x_loc = 31
    if y == "bottom":
        y_loc = 0
    elif y == "center":
        y_loc = 16
    elif y == "top":
        y_loc = 31
    if z == "bottom":
        z_loc = 0
    elif z == "center":
        z_loc = 16
    elif z == "top":
        z_loc = 31
    gt_arr[x_loc, y_loc, z_loc] = 1
    return gt_arr


def get_multi_point_gt() -> np.ndarray:
    gt_arr = np.zeros(shape=(32, 32, 32))
    gt_arr[16, 12:20, 16] = 1
    return gt_arr


@pytest.mark.parametrize("x_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("y_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("z_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("N", [1, 3, 5, 10, 20])
def test_single_point_get_pos_clicks2D_row_major(x_loc: locations, y_loc: locations, z_loc: locations, N: int):
    gt = get_single_point_gt(x_loc, y_loc, z_loc)
    # Make sure this does not crash.
    get_pos_clicks2D_row_major(gt, N)


@pytest.mark.parametrize("x_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("y_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("z_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("N", [1, 3, 5, 10, 20])
def test_single_point_point_interpolation(x_loc: locations, y_loc: locations, z_loc: locations, N: int):
    gt = get_single_point_gt(x_loc, y_loc, z_loc)
    points = point_interpolation(gt, N)


@pytest.mark.parametrize("x_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("y_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("z_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("N", [1, 3, 5, 10, 20])
def test_single_point_get_seed_point(x_loc: locations, y_loc: locations, z_loc: locations, N: int):
    gt = get_single_point_gt(x_loc, y_loc, z_loc)
    points = get_seed_point(gt, N, seed=10)


@pytest.mark.parametrize("x_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("y_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("z_loc", ["bottom", "center", "top"])
def test_single_point_get_minimal_boxes_row_major(x_loc: locations, y_loc: locations, z_loc: locations):
    gt = get_single_point_gt(x_loc, y_loc, z_loc)
    points = get_minimal_boxes_row_major(gt)


@pytest.mark.parametrize("x_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("y_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("z_loc", ["bottom", "center", "top"])
def test_single_point_get_bbox3d_sliced(x_loc: locations, y_loc: locations, z_loc: locations):
    gt = get_single_point_gt(x_loc, y_loc, z_loc)
    points = get_bbox3d_sliced(gt)


@pytest.mark.parametrize("x_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("y_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("z_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("N", [1, 3, 5, 10, 20])
def test_single_point_get_seed_boxes(x_loc: locations, y_loc: locations, z_loc: locations, N: int):
    gt = get_single_point_gt(x_loc, y_loc, z_loc)
    points = get_seed_boxes(gt, N)


@pytest.mark.parametrize("x_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("y_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("z_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("N", [1, 3, 5, 10, 20])
def test_single_point_box_interpolation(x_loc: locations, y_loc: locations, z_loc: locations, N: int):
    gt = get_single_point_gt(x_loc, y_loc, z_loc)
    box_seed_prompt = get_seed_boxes(gt, N)
    box_interpolation(box_seed_prompt)


@pytest.mark.parametrize("x_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("y_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("z_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("N", [1, 3, 5, 10, 20])
def test_single_point_get_pos_clicks3D(x_loc: locations, y_loc: locations, z_loc: locations, N: int):
    gt = get_single_point_gt(x_loc, y_loc, z_loc)
    points = get_pos_clicks3D(gt, N)


@pytest.mark.parametrize("x_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("y_loc", ["bottom", "center", "top"])
@pytest.mark.parametrize("z_loc", ["bottom", "center", "top"])
def test_single_point_get_pos_clicks3D(x_loc: locations, y_loc: locations, z_loc: locations):
    gt = get_single_point_gt(x_loc, y_loc, z_loc)
    points = get_bbox3d(gt)
