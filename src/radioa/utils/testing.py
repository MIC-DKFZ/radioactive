import numpy as np
import pickle


def gt_fg_dims(gt):
    """
    Get dimensions of bounding box of foreground : if more than 128x128x128, sammed3d will miss some
    """
    z, y, x = np.where(gt == 1)

    min_z, max_z = np.min(z), np.max(z)
    min_y, max_y = np.min(y), np.max(y)
    min_x, max_x = np.min(x), np.max(x)

    depth = max_z - min_z + 1
    height = max_y - min_y + 1
    width = max_x - min_x + 1

    return {"Depth": depth, "Height": height, "Width": width}


def test_save(obj, name):
    with open("/home/t722s/Desktop/test/" + name + ".pkl", "wb") as f:
        pickle.dump(obj, f)


def test_load(name):
    with open("/home/t722s/Desktop/test/" + name + ".pkl", "rb") as f:
        obj = pickle.load(f)
    return obj
