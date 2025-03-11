#%%
%matplotlib qt


# from openpiv.settings import PIVSettings

from matplotlib.patches import Polygon
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import pandas as pd 
import piv_processor as pp


plt.rcParams["image.cmap"] = "gray"

#%%


if __name__ == "__main__":
    # Example inputs
    Normalize_data = False
    base_filepath = os.path.join(os.getcwd(), "compression","20250205","RawVideos")
    subfolders = [os.path.join(base_filepath, o).replace('\\','/') for o in os.listdir(base_filepath) if os.path.isdir(os.path.join(base_filepath, o))]
    metadata = pd.read_excel(os.path.join(base_filepath.replace("RawVideos", "AuxiliaryFiles"), 'metadata.xlsx').replace('\\', '/'))
    # Create a dictionary to map exp_id to subfolder
    exp_id_to_subfolder = {i: subfolder for i, subfolder in enumerate(subfolders)}

    metadata['subfolder'] = metadata['exp_id'].map(exp_id_to_subfolder)

    # Save the updated DataFrame to a new file if needed
    metadata.to_csv(os.path.join(base_filepath.replace("RawVideos", "AuxiliaryFiles"), 'metadata_updated.csv'), index=False)

    for index, row in metadata.iterrows():
        subfolder = row.iloc[-1]
        paths = {
            "RawVideos": subfolder, 
            "Masked": subfolder.replace("RawVideos", "MaskedImages"),
            "Results": subfolder.replace("RawVideos", "Results"),
        }
        wall_bottom_left, wall_bottom_right, skip_value, image_seq, dt, v_max, sl = pp._return_metadata(pd.Series(row))
    
    #%%
    CF, baffle_edges = pp.calculate_baffle_dimensions(wall_bottom_left, wall_bottom_right)
    #wall_edges, wall_velocity = _mask_image_wall_th(paths["RawVideos"], paths["Masked"],baffle_edges, save_mask=True)
    
    wall_edges, wall_velocity = pp._mask_image_wall(paths["RawVideos"], paths["Masked"],baffle_edges,dt, CF, skip_frame=skip_value, save_mask=True)
    
    # Call the PIV analysis function
    #piv_analysis(paths["Masked"],paths['Results'], skip_value, image_seq, cpus=1)
    #%%
    results_file = Path(paths["Results"]).rglob("*.txt")
    x_grid, y_grid, u_grid, v_grid = pp.reconstruct_uv_grids(results_file, wall_edges, baffle_edges, wall_velocity, dt, CF, rank=10)
    v_mag, s11, s22, s33, w_vor = pp._return_fluid_derivative(x_grid, y_grid, u_grid, v_grid, wall_edges, baffle_edges,  CF)
    if Normalize_data:
        u_grid = u_grid / (v_max / 1000)
        v_grid = v_grid / (v_max / 1000)
        v_mag = v_mag / (v_max / 1000)
        w_vor = w_vor * (sl / v_max)
        s11 = s11 *(sl / v_max)
        s22 = s22 *(sl / v_max)
        s33 = s33 *(sl / v_max)
    pp.save_to_hdf5(x_grid, y_grid, u_grid, v_grid, v_mag, s11, s22, s33, w_vor,wall_edges, baffle_edges, paths)

    #%%
    ##  Plot single frame
    example_num = 200
    baffle_mask = pp._return_wall_baffle_mask(baffle_edges, x_grid, y_grid)
    wall_mask = pp._return_wall_baffle_mask(wall_edges[example_num], x_grid, y_grid)
    levels = np.linspace(
                -15, 15, 100
            )  #
   # Plot velocity magnitude with vectors
    fig, ax = plt.subplots(figsize=(5, 5))

    u_grid[example_num][baffle_mask], v_grid[example_num][baffle_mask] = np.nan, np.nan
    u_grid[example_num][~wall_mask], v_grid[example_num][~wall_mask] = np.nan, np.nan
     # Contour plot
    c = ax.contourf(x_grid, y_grid, w_vor[example_num], cmap="jet", levels=levels)
    cbar = fig.colorbar(c, ax=ax, label=r'vort', fraction=0.03)
    # Overlay wall edges
    ax.plot(
        wall_edges[example_num, :, 0], wall_edges[example_num, :, 1], "r", linewidth=2
    )

    # Velocity vectors with quiver
    ax.quiver(x_grid, y_grid, u_grid[example_num], v_grid[example_num], color="k", linewidth=1, scale=0.5)
    # Overlay filled baffle edges
    baffle_polygon = Polygon(baffle_edges, closed=True, color="gray", alpha=0.6)
    ax.add_patch(baffle_polygon)

    # Adjust plot appearance
    ax.invert_yaxis()
    ax.set_xlim([x_grid.min(), x_grid.max()])
    ax.set_ylim([y_grid.max() - 300, y_grid.min() + 200])
    ax.set_aspect("equal")  # Ensure equal aspect ratio
    ax.axis("off")

    plt.tight_layout()
    plt.show()
    #%%
    #  Plot a video 
   
    results_file = Path(paths["Results"]).rglob("*.txt")
    pp.create_video_from_images(
        wall_edges, baffle_edges, results_file, paths['Results'],wall_velocity, v_max=v_max, sl=sl, Normalize_data=True,dt=dt, CF=CF  # # Set the variable to plot (v_mag, w_vor, or s33)
    )

# %%
