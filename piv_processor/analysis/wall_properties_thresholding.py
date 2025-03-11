# %%

import matplotlib.pyplot as plt

plt.rcParams["image.cmap"] = "gray"
import numpy as np
from pathlib import Path
from skimage.io import imread, imsave
from skimage.segmentation import slic
from skimage.color import rgb2gray
from scipy.ndimage import uniform_filter1d
import cv2
import tqdm
from scipy.interpolate import interp1d

# %%


def smooth_contour(contour, window_size=5, fixed_length=100):
    smoothed_contour_x = uniform_filter1d(contour[:, 0], size=window_size)
    smoothed_contour_y = uniform_filter1d(contour[:, 1], size=window_size)
    smoothed_contour = np.stack((smoothed_contour_x, smoothed_contour_y), axis=-1)

    # Interpolate to fixed length
    if len(smoothed_contour) < fixed_length:
        padding = np.zeros((fixed_length - len(smoothed_contour), 2))
        smoothed_contour = np.vstack((smoothed_contour, padding))
    else:
        interp_x = interp1d(
            np.linspace(0, 1, len(smoothed_contour)),
            smoothed_contour[:, 0],
            kind="linear",
        )
        interp_y = interp1d(
            np.linspace(0, 1, len(smoothed_contour)),
            smoothed_contour[:, 1],
            kind="linear",
        )
        smoothed_contour_x = interp_x(np.linspace(0, 1, fixed_length))
        smoothed_contour_y = interp_y(np.linspace(0, 1, fixed_length))
        smoothed_contour = np.stack((smoothed_contour_x, smoothed_contour_y), axis=-1)

    return smoothed_contour


def _mask_image_wall_th(
    filepath,
    filepath_masked,
    baffle_edges,
    save_mask=False,
    ds=3,
):

    valid_extensions = {".png", ".tif", ".jpeg", ".jpg"}

    # Get all files with valid extensions
    image_files = sorted(
        [
            file
            for file in Path(filepath).iterdir()
            if file.suffix.lower() in valid_extensions
        ]
    )

    masked_images_folder = Path(filepath_masked)
    masked_images_folder.mkdir(exist_ok=True, parents=True)

    wall_velocity = np.zeros([len(image_files), 5, 2])
    final_instances_closed = []
    struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ds, ds))
    print("Start: masking")
    for file in tqdm.tqdm(image_files, total=len(image_files)):
        image = imread(file)
        th_image = slic(image, n_segments=2, compactness=20).astype(np.uint8) - 1

        contours, _ = cv2.findContours(
            th_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )[-2:]
        # Filter contours with max length
        contours = np.squeeze(max(contours, key=len))
        smoothed_contour = smooth_contour(contours, 10)
        smoothed_contour[-1] = smoothed_contour[0]

        masked_wall = cv2.dilate(th_image, struct_element, iterations=3)

        # Fill the polygon with 0 (black) on the mask
        cv2.fillPoly(masked_wall, [baffle_edges], 0)
        masked_wall = masked_wall * rgb2gray(image)
        final_instances_closed.append(smoothed_contour)
        if save_mask:
            # Save the masked image
            masked_image_path = masked_images_folder.joinpath(file.name)
            imsave(masked_image_path, masked_wall)
    final_instances_closed = np.array(final_instances_closed).astype(np.float32)
    wall_velocity = np.zeros([final_instances_closed.shape[0], 5, 2])
    print("Finished: Masking")
    return final_instances_closed, wall_velocity


# %%
if __name__ == "__main__":
    filepath = (
        r"C:\Users\WeheliyeWeheliye\OneDrive - Oribiotech Ltd\Desktop\Ori_Weheliye\PIV\PIV_Ori_github\piv_process\compression\20250205\RawVideos\exp_id_0_ar_13_bh_6"
    ).replace("\\", "/")
    filepath_masked = filepath.replace("RawVideos", "MaskedImages_th")
    baffle_edges = np.array(
        [
            [145, 0],
            [290, 0],
            [290, 500],
            [749, 500],
            [749, 519],
            [0, 519],
            [0, 500],
            [145, 500],
            [145, 0],
        ]
    )

# %%
