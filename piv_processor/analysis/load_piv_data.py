# %%


# from openpiv.settings import PIVSettings


import numpy as np

import matplotlib.pyplot as plt

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


def open_piv_data(filename, baffle_edges, wall_edges, file_id=0, dt=5, CF=15 / 160):
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

    v_magnitude = np.sqrt(u_grid**2 + v_grid**2)

    delta_x = np.mean(np.diff(unique_x)) * (
        CF / 1000
    )  # Spacing in x-direction in meters (mm to m)
    delta_y = np.mean(np.diff(unique_y)) * (
        CF / 1000
    )  # Spacing in y-direction in meters (mm to m)

    # Initialize vorticity array
    vorticity_curl = np.zeros(grid_shape)
    shear_curl = np.zeros(grid_shape)

    # Loop through the grid and calculate vorticity at each point
    for i in range(1, grid_shape[0] - 1):  # skip boundaries
        for j in range(1, grid_shape[1] - 1):  # skip boundaries
            # Compute partial derivatives using central difference
            du_dx = (u_grid[i, j + 1] - u_grid[i, j - 1]) / (2 * delta_x)  # ∂u/∂x
            dv_dy = (v_grid[i + 1, j] - v_grid[i - 1, j]) / (2 * delta_y)  # ∂v/∂y
            du_dy = (u_grid[i + 1, j] - u_grid[i - 1, j]) / (2 * delta_y)  # ∂u/∂y
            dv_dx = (v_grid[i, j + 1] - v_grid[i, j - 1]) / (2 * delta_x)  # ∂v/∂x

            # Calculate vorticity (ω = ∂u/∂y - ∂v/∂x)
            vorticity_curl[i, j] = du_dy - dv_dx
            shear_curl[i, j] = du_dy + dv_dx

    # Mask velocities outside the wall
    v_magnitude[~wall_mask], vorticity_curl[~wall_mask], shear_curl[~wall_mask] = (
        0,
        0,
        0,
    )
    u_grid[baffle_mask], v_grid[baffle_mask], u_grid[~wall_mask], v_grid[~wall_mask] = (
        0,
        0,
        0,
        0,
    )

    return x_grid, y_grid, u_grid, v_grid
