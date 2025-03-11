# %%
# %matplotlib qt
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

from openpiv import windef  # <---- see windef.py for details
# from openpiv.settings import PIVSettings
from matplotlib.patches import Polygon
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import os
from piv_processor.analysis.load_piv_data import _return_wall_baffle_mask
from piv_processor.analysis.fluid_velocity_derivatives import reconstruct_uv_grids, _return_fluid_derivative
plt.rcParams["image.cmap"] = "gray"
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


def get_data_to_plot(
    plot_variable, image_id, Normalize_data, v_mag, w_vor, s33, sl, v_max
):
    if plot_variable == "v_mag":
        data_to_plot = v_mag[image_id]
        colorbar_label = r"$V/V_{max} [-]$" if Normalize_data else r"V [m/s]"
        cmap = "jet"
        q_color = "w"
        levels = np.linspace(np.nanmin(v_mag), np.nanmax(v_mag), 100)
    elif plot_variable == "w_vor":
        data_to_plot = w_vor[image_id]
        levels = (
            np.linspace(-15 * (sl / v_max), 15 * (sl / v_max), 100)
            if Normalize_data
            else np.linspace(-15, 15, 100)
        )
        cmap = "bwr"
        q_color = "k"
        colorbar_label = (
            r"$\omega \cdot \left(\frac{sl}{V_{max}}\right) [-]$"
            if Normalize_data
            else r"$\omega [1/s]$"
        )
    elif plot_variable == "s33":
        data_to_plot = s33[image_id]
        levels = np.linspace(np.nanmin(s33), np.nanmax(s33), 100)
        cmap = "jet"
        q_color = "w"
        colorbar_label = (
            r"$\tau_{max} \cdot \left(\frac{sl}{V_{max}}\right) [-]$"
            if Normalize_data
            else r"$\tau_{max} [1/s]$"
        )
    else:
        raise ValueError("Invalid plot_variable. Choose 'v_mag', 'w_vor', or 's33'.")

    # levels = np.linspace(data_to_plot.min(), data_to_plot.max(), 100)
    return data_to_plot, colorbar_label, levels, cmap, q_color


def _create_output_video_folder(output_video_folder):
    output_video_folder = Path(output_video_folder)
    output_video_folder.mkdir(exist_ok=True, parents=True)


def create_video_from_images(
    wall_edges,
    baffle_edges,
    results_file,
    results_folder,
    wall_velocity,
    v_max,
    sl,
    Normalize_data=True,
    dt=5,
    CF=15 / 160,
    skip_rate=10,
):

    x_grid, y_grid, u_grid, v_grid = reconstruct_uv_grids(
        results_file, wall_edges, baffle_edges, wall_velocity, dt, CF, rank=10
    )
    v_mag, s11, s22, s33, w_vor = _return_fluid_derivative(
        x_grid,
        y_grid,
        u_grid,
        v_grid,
        wall_edges,
        baffle_edges,
        CF,
    )
    if Normalize_data:
        u_grid = u_grid / (v_max / 1000)
        v_grid = v_grid / (v_max / 1000)
        v_mag = v_mag / (v_max / 1000)
        w_vor = w_vor * (sl / v_max)
        s11 = s11 * (sl / v_max)
        s22 = s22 * (sl / v_max)
        s33 = s33 * (sl / v_max)

    baffle_mask = _return_wall_baffle_mask(baffle_edges, x_grid, y_grid)

    # Set the variable to plot (v_mag, w_vor, or s33)

    for image_id in range(0, u_grid.shape[0], skip_rate):
        wall_mask = _return_wall_baffle_mask(wall_edges[image_id], x_grid, y_grid)

        # Mask out regions for visualization
        u_grid[image_id][baffle_mask], v_grid[image_id][baffle_mask] = np.nan, np.nan
        u_grid[image_id][~wall_mask], v_grid[image_id][~wall_mask] = np.nan, np.nan
        # Get data to plot
        for plot_variable in ["v_mag", "w_vor", "s33"]:
            output_video_path = os.path.join(
                results_folder,
                f"output_video_instant_velocity_{plot_variable}_{Normalize_data}",
            ).replace("\\", "/")
            _create_output_video_folder(output_video_path)
            data_to_plot, colorbar_label, levels, cmap, q_color = get_data_to_plot(
                plot_variable, image_id, Normalize_data, v_mag, w_vor, s33, sl, v_max
            )

            # Create the plot
            fig, ax = plt.subplots(figsize=(5, 5))

            # Contour plot
            c = ax.contourf(x_grid, y_grid, data_to_plot, cmap=cmap, levels=levels)
            cbar = fig.colorbar(c, ax=ax, label=colorbar_label, fraction=0.03)

            # Overlay filled baffle edges
            baffle_polygon = Polygon(baffle_edges, closed=True, color="gray", alpha=0.6)
            ax.add_patch(baffle_polygon)

            # Overlay wall edges
            ax.plot(
                wall_edges[image_id, :, 0], wall_edges[image_id, :, 1], "r", linewidth=2
            )

            # Velocity vectors with quiver
            if Normalize_data:
                ax.quiver(
                    x_grid,
                    y_grid,
                    u_grid[image_id],
                    v_grid[image_id],
                    color=q_color,
                    linewidth=1,
                )
            else:
                ax.quiver(
                    x_grid,
                    y_grid,
                    u_grid[image_id],
                    v_grid[image_id],
                    color=q_color,
                    linewidth=1,
                    scale=0.5,
                )

                # Add text in the top right corner
            ax.text(
                0.95,
                0.95,
                f"$t = {image_id*5} ms$",
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
            )

            # Adjust plot appearance
            ax.invert_yaxis()
            ax.set_xlim([x_grid.min(), x_grid.max()])
            ax.set_ylim([y_grid.max() - 300, y_grid.min() + 200])
            ax.set_aspect("equal")
            ax.axis("off")

            plt.tight_layout()
            plt.show()

            # Save the figure as a frame
            frame_path = Path(output_video_path) / f"frame_{image_id:05}.png"
            plt.savefig(frame_path, dpi=500, bbox_inches="tight")
            plt.close(fig)
