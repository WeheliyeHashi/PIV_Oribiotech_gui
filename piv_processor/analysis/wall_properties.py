# %%

from matplotlib.patches import Polygon
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt

from imageio import imread


from skimage.color import rgb2gray
from skimage.io import imread, imsave

import sleap
from scipy.interpolate import interp1d

from scipy.ndimage import uniform_filter1d
import cv2


# %%


def _return_wall_baffle_mask(baffle_edges, x_grid, y_grid):
    # Initialize the mask
    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    # Use cv2.pointPolygonTest to check if points are inside the polygon
    inside = np.array(
        [
            cv2.pointPolygonTest(baffle_edges, tuple(pt), measureDist=False) >= 0
            for pt in points
        ]
    )

    # Reshape the result back to the shape of the grid
    baffle_mask = inside.reshape(x_grid.shape)

    return baffle_mask


def _instances(label, image_files, skip_frame=4):
    # Initialize final_instances with NaNs
    final_instances = np.full((len(image_files), 6, 2), np.nan)

    for i, labeled_frame in enumerate(label.labels):
        for j, instance in enumerate(labeled_frame.instances):
            # Assuming instance has x and y coordinates
            x_coords = [point.x for point in instance.points if instance.score > 0.5]
            y_coords = [point.y for point in instance.points if instance.score > 0.5]

            if x_coords and y_coords:
                for k in range(
                    min(len(x_coords), 6)
                ):  # Ensure we don't exceed the second dimension size
                    final_instances[i * skip_frame, k, 0] = x_coords[k]
                    final_instances[i * skip_frame, k, 1] = y_coords[k]

    # Handle NaNs before applying uniform_filter1d
    for i in range(6):
        valid_mask = ~np.isnan(final_instances[:, i, 1])
        final_instances[valid_mask, i, 1] = uniform_filter1d(
            final_instances[valid_mask, i, 1], size=20
        )

    final_instances[:, ::2, 0] = np.nanmedian(final_instances[:, ::2, 0])
    final_instances[:, 1::2, 0] = np.nanmedian(final_instances[:, 1::2, 0])
    # Interpolate the skipped frames
    for k in range(6):
        for dim in range(2):
            # Get the indices and values of the non-skipped frames
            indices = np.arange(0, len(image_files), skip_frame)
            values = final_instances[indices, k, dim]

            # Remove NaNs from values and corresponding indices
            valid = ~np.isnan(values)
            indices = indices[valid]
            values = values[valid]

            # Interpolate
            f = interp1d(indices, values, kind="linear", fill_value="extrapolate")
            final_instances[:, k, dim] = f(np.arange(len(image_files)))

    return final_instances


def _return_wall_velocity(final_instances, dt, CF, rank=2, plot=False):
    # Initialize the wall velocity array
    wall_velocity = np.zeros_like(final_instances)
    wall_velocity_smoothed = np.zeros_like(final_instances)

    # Loop through the instances
    for i in range(final_instances.shape[1]):
        # Compute the velocity of the wall
        wall_velocity[:, i, 1] = uniform_filter1d(
            np.gradient(final_instances[:, i, 1], dt) * CF, size=20
        )

    # Perform SVD
    U, S, Vt = np.linalg.svd(wall_velocity[:, :, 1], full_matrices=False)

    # Reconstruct the wall velocity using the first 2 ranks
    S_reduced = np.zeros_like(S)
    S_reduced[:rank] = S[:rank]
    wall_velocity_reconstructed = np.dot(U, np.dot(np.diag(S_reduced), Vt))

    # Update the wall_velocity with the reconstructed values
    wall_velocity_smoothed[:, :, 1] = wall_velocity_reconstructed

    if plot:
        colors = ["r", "b", "g", "c", "m", "y"]  # Define a list of colors for the plots

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot individual points on the first subplot
        for i in range(wall_velocity_smoothed.shape[1]):
            ax1.plot(wall_velocity_smoothed[:, i, 1], colors[i], label=f"Point {i}")

        ax1.set_xlabel("time", fontsize=24)
        ax1.set_ylabel("Velocity [m/s]", fontsize=24)
        # ax1.set_title("Wall Velocity")
        ax1.legend()

        # Calculate the nanmedian of each point
        nanmedian_values = np.nanmedian(wall_velocity_smoothed[:, :, 1], axis=0)

        # Plot the nanmedian values on the second subplot
        ax2.plot(range(6), nanmedian_values, "o-", label="Nanmedian", color="k")

        # Set x and y labels with font size 24
        ax2.set_xlabel("Wall points", fontsize=24)

        ax2.set_ylabel(r"$\overline{V_w}$ [m/s]", fontsize=24)

        # Set x-tick and y-tick label font size to 24
        ax1.tick_params(axis="both", which="major", labelsize=18)
        ax2.tick_params(axis="both", which="major", labelsize=18)

        # ax2.legend()

        # Adjust layout
        plt.tight_layout()
        plt.show()

    return wall_velocity_smoothed


def _mask_image_wall(
    filepath,
    filepath_masked,
    model_path,
    baffle_edges,
    dt=5,
    CF=15 / 160,
    skip_frame=1,
    save_mask=False,
):
    valid_extensions = {".png", ".tif", ".jpeg", ".jpg"}

    # Get all files with valid extensions
    image_files = [
        file
        for file in Path(filepath).iterdir()
        if file.suffix.lower() in valid_extensions
    ]

    masked_images_folder = Path(filepath_masked)
    masked_images_folder.mkdir(exist_ok=True, parents=True)

     # Locate the subfolders and assign them to MODEL_CENTROID and MODEL_INSTANCE
    model_subfolders = list(Path(model_path).glob("*"))
    model_centroid = next((str(folder) for folder in model_subfolders if "centroid" in folder.name), None)
    model_instance = next((str(folder) for folder in model_subfolders if "instance" in folder.name), None)

    if not model_centroid or not model_instance:
        raise FileNotFoundError("Required model subfolders not found in the specified model path.")

    MODEL_CENTROID = os.path.join(model_centroid)
    MODEL_INSTANCE = os.path.join(model_instance)
    #MODEL_CENTROID = os.path.join("models/250120_134903.centroid.n=34")
    #MODEL_INSTANCE = os.path.join("models/250120_135829.centered_instance.n=34")

    predictor = sleap.load_model([MODEL_CENTROID, MODEL_INSTANCE], batch_size=2)
    images = [imread(file) for file in image_files[::skip_frame]]
    prediction = predictor.predict(np.array(images))
    final_instances = _instances(prediction, image_files, skip_frame=skip_frame)
    final_instances_closed = []

    for id_file, (file, contour) in enumerate(zip(image_files, final_instances)):
        mask = np.zeros(images[0].shape[:2], dtype=np.uint8)  # Ensure mask is 2D

        max_y = np.max(contour[:, 1])
        min_y = np.min(contour[:, 1])

        # Find the first non-NaN point
        first_non_nan_point = contour[~np.isnan(contour).any(axis=1)][0]
        new_points = np.array(
            [
                [0, min_y],
                [0, max_y],
                first_non_nan_point,  # Add the first point to close the polygon
            ]
        )

        # Append new points to the contour
        closed_contour = np.vstack([contour, new_points]).astype(
            np.int32
        )  # Ensure the contour is of type int32

        # Draw the contour on the mask
        if save_mask:
            masked_wall = cv2.drawContours(
                mask, [closed_contour], -1, (1), thickness=cv2.FILLED
            )
            cv2.fillPoly(masked_wall, [baffle_edges], 0)
            masked_wall = masked_wall * rgb2gray(imread(image_files[id_file]))

            # Save the masked image
            masked_image_path = masked_images_folder.joinpath(file.name)
            imsave(masked_image_path, masked_wall)
        final_instances_closed.append(closed_contour)

    final_instances_closed = np.array(final_instances_closed)

    wall_velocity = _return_wall_velocity(final_instances, dt, CF)

    return final_instances_closed, wall_velocity
