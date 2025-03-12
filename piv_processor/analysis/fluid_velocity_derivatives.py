

# from openpiv.settings import PIVSettings

from matplotlib.patches import Polygon
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from numpy.linalg import eig


import cv2
from shapely.geometry import LineString, Point
from scipy.interpolate import griddata


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


def open_piv_data(filename, wall_edges, baffle_edges, file_id=0, dt=5, CF=15 / 160):
    # Load the data from the text file
    data = np.loadtxt(filename, skiprows=1)  # Skip the header row

    # Extract columns
    x = data[:, 0]
    y = data[:, 1]
    u = data[:, 2]
    v = data[:, 3]
    mask = data[:, 4]

    # Determine the unique grid dimensions
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    grid_shape = (len(unique_y), len(unique_x))

    # Reshape the arrays into 2D grids

    x_grid = x.reshape(grid_shape)
    y_grid = y.reshape(grid_shape)
    u_grid = u.reshape(grid_shape)
    v_grid = v.reshape(grid_shape)

    u_grid = np.flipud(u_grid) * CF / dt
    v_grid = np.flipud(v_grid) * CF / dt

    # Apply masks to the velocity fields
    baffle_mask = _return_wall_baffle_mask(baffle_edges, x_grid, y_grid)
    wall_mask = _return_wall_baffle_mask(wall_edges[file_id], x_grid, y_grid)

    # Mask velocities inside the baffle
    u_grid[baffle_mask], v_grid[baffle_mask] = 0, 0

    # Mask velocities outside the wall
    u_grid[~wall_mask], v_grid[~wall_mask] = np.nan, np.nan

    u_grid[baffle_mask], v_grid[baffle_mask], u_grid[~wall_mask], v_grid[~wall_mask] = (
        0,
        0,
        0,
        0,
    )

    return x_grid, y_grid, u_grid, v_grid


def compute_gradients(u, v, dx, dy):
    """
    Compute velocity gradients using central differences.
    """
    dudx = np.gradient(u, dx, axis=1)
    dudy = np.gradient(u, dy, axis=0)
    dvdx = np.gradient(v, dx, axis=1)
    dvdy = np.gradient(v, dy, axis=0)
    return dudx, dudy, dvdx, dvdy


def _compute_fluid_derivative(x, y, u, v, baffle_mask, wall_mask, CF):
    # Determine the unique grid dimensions
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    grid_shape = (len(unique_y), len(unique_x))

    delta_x = np.mean(np.diff(unique_x)) * (
        CF / 1000
    )  # Spacing in x-direction in meters (mm to m)

    delta_y = np.mean(np.diff(unique_y)) * (
        CF / 1000
    )  # Spacing in y-direction in meters (mm to m)

    """
    Compute the lambda-2 field for a 2D velocity field.
    """
    # Compute velocity gradients
    dudx, dudy, dvdx, dvdy = compute_gradients(u, v, delta_x, delta_y)

    # Dimensions of the field
    ny, nx = u.shape
    S11 = np.zeros_like(u)
    S22 = np.zeros_like(u)
    S33 = np.zeros_like(u)
    w_l = np.zeros_like(u)
    vorticity_curl = dvdx - dudy
    v_magnitude = np.sqrt(u**2 + v**2)

    # Loop over the grid points to compute shear rates
    for i in range(ny):
        for j in range(nx):
            # Rate-of-strain tensor (S) with the eigenvalues of the tensor. Principal strain rates.
            S = 0.5 * np.array(
                [
                    [dudx[i, j] + dudx[i, j], dudy[i, j] + dvdx[i, j]],
                    [dudy[i, j] + dvdx[i, j], dvdy[i, j] + dvdy[i, j]],
                ]
            )
            S11[i, j] = np.max(eig(S)[0])
            S22[i, j] = np.min(eig(S)[0])
            S33[i, j] = (S11[i, j] - S22[i, j]) / 2

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            localcirculation = (
                -0.5 * delta_x * (u[i + 1, j + 1] + 2 * u[i + 1, j] + u[i + 1, j - 1])
                - 0.5 * delta_y * (v[i - 1, j - 1] + 2 * v[i, j - 1] + v[i + 1, j - 1])
                + 0.5 * delta_x * (u[i - 1, j - 1] + 2 * u[i - 1, j] + u[i - 1, j + 1])
                + 0.5 * delta_y * (v[i - 1, j + 1] + 2 * v[i, j + 1] + v[i + 1, j + 1])
            )

            w_l[i, j] = localcirculation / (4 * delta_x * delta_y)

    vorticity_curl[1:-1, 1:-1] = w_l[1:-1, 1:-1]
    # Mask velocities outside the wall

    (
        v_magnitude[~wall_mask],
        S11[~wall_mask],
        S22[~wall_mask],
        S33[~wall_mask],
        vorticity_curl[~wall_mask],
    ) = (
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    )  # (0, 0, 0, 0, 0)

    return v_magnitude, S11, S22, S33, vorticity_curl


def _return_fluid_derivative(
    x_grid, y_grid, u_grid, v_grid, wall_edges, baffle_edges, CF
):
    """_summary_

    Args:
        x_grid (_array_): x grid in radial space
        y_grid (array): y grid in axial space
        u_grid (array): radial velocity grid
        v_grid (array): axial velcoity grid
        baffle_mask (array): baffle edges
        wall_mask (array): wall edges
        CF (array): Calibration factor mm/pixel
    Returns:
        grid array: fluid derivatives
    """
    # Compute the fluid derivatives
    print("Start: Compute the fluid derivatives")
    v_mag = []
    s11 = []
    s22 = []
    s33 = []
    w_vor = []
    baffle_mask = _return_wall_baffle_mask(baffle_edges, x_grid, y_grid)
    for file_id in range(u_grid.shape[0]):

        wall_mask = _return_wall_baffle_mask(wall_edges[file_id], x_grid, y_grid)
        v_magnitude, S11, S22, S33, w = _compute_fluid_derivative(
            x_grid,
            y_grid,
            u_grid[file_id],
            v_grid[file_id],
            baffle_mask,
            wall_mask,
            CF,
        )
        v_mag.append(v_magnitude)
        s11.append(S11)
        s22.append(S22)
        s33.append(S33)
        w_vor.append(w)
    v_mag = np.array(v_mag)
    s11 = np.array(s11)
    s22 = np.array(s22)
    s33 = np.array(s33)
    w_vor = np.array(w_vor)

    print("Finish: Calculating the fluid derivatives")

    return v_mag, s11, s22, s33, w_vor


def single_time_step_wall_velocity_analysis(
    wall_edges, wall_velocity, x_grid, y_grid, time_stamp
):

    # Initialize the combined velocity grids
    mag_values = np.append(
        wall_velocity[time_stamp, 0, 1], wall_velocity[time_stamp, :, 1]
    )
    y_values = np.append(
        wall_edges[time_stamp, 0, 1],
        wall_edges[time_stamp, : wall_velocity.shape[1], 1],
    )
    x_values = np.append(0, wall_edges[time_stamp, : wall_velocity.shape[1], 0])

    # Create meshgrid and combine into grid points
    grid_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    line = LineString(np.column_stack((x_values, y_values)))

    # Corrected: Compute distances in a vectorized manner
    distances = np.array([line.distance(Point(p)) for p in grid_points]).reshape(
        x_grid.shape
    )

    # Interpolate wall velocities and handle NaNs
    new_wall_velocity = np.nan_to_num(
        griddata((x_values, y_values), mag_values, (x_grid, y_grid), method="linear")
    )

    tolerance = 16  # Define tolerance for points near the line
    wall_con_edges = np.zeros_like(x_grid)

    # Assign velocities to points within the tolerance
    mask = distances <= tolerance
    wall_con_edges[mask] = new_wall_velocity[mask]
    return wall_con_edges * -1, mask

    # plt.contourf(x_grid, y_grid, wall_con_edges, cmap='jet')


def plot_velocity(
    x_grid,
    y_grid,
    data,
    title,
    baffle_edges,
    wall_edges,
    time_stamp,
    colorbar_label,
    save_path,
):
    """
    Plots a velocity field with baffle and wall overlays and saves the figure.

    Parameters:
        x_grid, y_grid: 2D arrays defining the grid.
        data: 2D array of the velocity field to plot.
        title: Title of the plot.
        baffle_edges: Coordinates of baffle edges.
        wall_edges: 3D array with wall edge coordinates over time.
        time_stamp: Time index for wall_edges.
        colorbar_label: Label for the colorbar.
        save_path: Path to save the plot image.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Contour plot
    levels = np.linspace(data.min(), data.max(), 100)
    c = ax.contourf(x_grid, y_grid, data, levels=levels, cmap="jet")

    # Add baffle and wall overlays
    ax.add_patch(Polygon(baffle_edges, closed=True, color="gray", alpha=1))
    ax.plot(
        wall_edges[time_stamp, :, 0], wall_edges[time_stamp, :, 1], "w", linewidth=2
    )

    # Axis settings
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_xlim([x_grid.min(), x_grid.max()])
    ax.set_ylim([y_grid.max() - 300, y_grid.min() + 200])
    ax.set_aspect("equal")
    ax.axis("off")

    # Colorbar
    fig.colorbar(c, ax=ax, label=colorbar_label, fraction=0.03, pad=0.04)

    # Save and show
    fig.tight_layout()
    fig.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.show()


def _combine_fluid_wall_velocity(
    u_grid,
    v_grid,
    x_grid,
    y_grid,
    baffle_edges,
    wall_edges,
    wall_velocity,
    time_stamp,
    plot=False,
):
    # Initialize the combined velocity grids

    # Loop through the grid points
    v_grid_updated = v_grid.copy()

    wall_instant_velo, mask = single_time_step_wall_velocity_analysis(
        wall_edges, wall_velocity, x_grid, y_grid, time_stamp
    )
    #  Assign the boundary conditions
    v_grid[mask != 0] = 0
    v_grid_updated[mask != 0] = wall_instant_velo[mask != 0]
    u_grid[mask != 0] = 0

    if plot:

        # Usage examples
        plot_velocity(
            x_grid,
            y_grid,
            v_grid,
            title="Axial Fluid Velocity",
            baffle_edges=baffle_edges,
            wall_edges=wall_edges,
            time_stamp=time_stamp,
            colorbar_label="Velocity Magnitude [m/s]",
            save_path="axial_fluid_velocity.png",
        )

        plot_velocity(
            x_grid,
            y_grid,
            wall_instant_velo,
            title="Wall Velocity",
            baffle_edges=baffle_edges,
            wall_edges=wall_edges,
            time_stamp=time_stamp,
            colorbar_label="Velocity Magnitude [m/s]",
            save_path="wall_velocity.png",
        )

        plot_velocity(
            x_grid,
            y_grid,
            v_grid_updated,
            title="Combined Fluid and Wall Velocity",
            baffle_edges=baffle_edges,
            wall_edges=wall_edges,
            time_stamp=time_stamp,
            colorbar_label="Velocity Magnitude [m/s]",
            save_path="combined_velocity.png",
        )

    return u_grid, v_grid_updated


def reconstruct_uv_grids(
    results_files, wall_edges, baffle_edges, wall_velocity, dt, CF, rank=2
):
    print("Start: Phase averaging the flow")
    u_grids = []
    v_grids = []

    for image_id, result_file in zip(range(wall_edges.shape[0]), results_files):

        x_grid, y_grid, u_grid, v_grid = open_piv_data(
            result_file,
            wall_edges,
            baffle_edges,
            file_id=image_id,
            dt=dt,
            CF=CF,
        )
        u_grid, v_grid = _combine_fluid_wall_velocity(
            u_grid,
            v_grid,
            x_grid,
            y_grid,
            baffle_edges,
            wall_edges,
            wall_velocity,
            image_id,
        )

        u_grids.append(u_grid)
        v_grids.append(v_grid)

    # Convert lists to numpy arrays
    u_grids = np.array(u_grids)
    v_grids = np.array(v_grids)

    # Flatten u_grids and v_grids and stack them vertically
    combined_grids = np.vstack(
        (u_grids.reshape(u_grids.shape[0], -1), v_grids.reshape(v_grids.shape[0], -1))
    )

    # Perform SVD on the combined grids
    u, s, vh = np.linalg.svd(combined_grids, full_matrices=False)

    # Reconstruct the combined grids using the specified rank
    s_reduced = np.diag(s[:rank])
    u_reduced = u[:, :rank]
    vh_reduced = vh[:rank, :]
    reconstructed_combined_grids = np.dot(u_reduced, np.dot(s_reduced, vh_reduced))

    # Split the reconstructed combined grids back into u_grids and v_grids
    reconstructed_u_grids = reconstructed_combined_grids[: u_grids.shape[0], :].reshape(
        u_grids.shape
    )
    reconstructed_v_grids = reconstructed_combined_grids[u_grids.shape[0] :, :].reshape(
        v_grids.shape
    )
    print("Finish: Phase averaging the flow")
    return x_grid, y_grid, reconstructed_u_grids, reconstructed_v_grids
