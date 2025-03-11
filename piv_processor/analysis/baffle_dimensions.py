# %%
from openpiv import windef  # <---- see windef.py for details

# from openpiv.settings import PIVSettings

from matplotlib.patches import Polygon
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from numpy.linalg import eig
from imageio import imread, get_writer

from skimage.color import rgb2gray
from skimage.io import imread, imsave
from scipy.signal import savgol_filter
import sleap
from scipy.interpolate import interp1d

plt.rcParams["image.cmap"] = "gray"
from scipy.ndimage import uniform_filter1d
import cv2


# %%
def calculate_baffle_dimensions(
    wall_bottom_left: np.ndarray, wall_bottom_right: np.ndarray
):
    CF = 15 / (wall_bottom_right[0] - wall_bottom_left[0])  # conversion factor mm/pixel
    Diameter = 110 / CF
    Baffle_thickness = 2 / CF
    baffle_edges = np.array(
        [
            [wall_bottom_left[0], 0],
            [wall_bottom_right[0], 0],
            wall_bottom_right,
            [
                (wall_bottom_right[0] + wall_bottom_left[0]) / 2 + Diameter / 2,
                wall_bottom_right[1],
            ],
            [
                (wall_bottom_right[0] + wall_bottom_left[0]) / 2 + Diameter / 2,
                wall_bottom_right[1] + Baffle_thickness,
            ],
            [0, wall_bottom_left[1] + Baffle_thickness],
            [0, wall_bottom_left[1]],
            wall_bottom_left,
            [wall_bottom_left[0], 0],
        ],
        dtype=np.int32,
    )

    return CF, baffle_edges
