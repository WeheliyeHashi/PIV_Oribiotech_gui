#%%
# from openpiv.settings import PIVSettings

from matplotlib.patches import Polygon
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import pandas as pd
import piv_processor as pp

import argparse
plt.rcParams["image.cmap"] = "gray"

#%%
def main_processor(base_filepath, model_path, Normalize_data=True, n_cpus=4, plot_video=True,  metadata_file=None):

    """
    Main function to process PIV data.
    Args:
        base_filepath (str): Base file path for raw videos
        model_path (str): Path to the model.
        Normalize_data (bool): Normalize data flag.
        n_cpus (int): Number of CPUs to use.
        plot_video (bool): Flag to plot video.
        metadata_file (str): Path to metadata file.
    """
    # Get the subfolders in the base file path
    subfolders = [os.path.join(base_filepath, o).replace('\\','/') for o in os.listdir(base_filepath) if os.path.isdir(os.path.join(base_filepath, o))]
    metadata = pd.read_excel(metadata_file or os.path.join(base_filepath.replace("RawVideos", "AuxiliaryFiles"), 'metadata.xlsx').replace('\\', '/'))
    # Create a dictionary to map exp_id to subfolder
    exp_id_to_subfolder = {i: subfolder for i, subfolder in enumerate(subfolders)}

    metadata['subfolder'] = metadata['exp_id'].map(exp_id_to_subfolder)

    # Save the updated DataFrame to a new file if needed
    updated_metadata_path = os.path.join(base_filepath.replace("RawVideos", "AuxiliaryFiles"), 'metadata_updated.csv')
    if os.path.exists(updated_metadata_path):
        os.remove(updated_metadata_path)
    metadata.to_csv(updated_metadata_path, index=False)
    total_files = len(metadata)
    for index, row in metadata.iterrows():
        subfolder = row.iloc[-1]
        print(f"Processing file {index + 1} of {total_files}: {subfolder}")   
        paths = {
            "RawVideos": subfolder, 
            "Masked": subfolder.replace("RawVideos", "MaskedImages"),
            "Results": subfolder.replace("RawVideos", "Results"),
        }
        wall_bottom_left, wall_bottom_right, skip_value, image_seq, dt, v_max, sl = pp._return_metadata(pd.Series(row))
    
        CF, baffle_edges = pp.calculate_baffle_dimensions(wall_bottom_left, wall_bottom_right)
        
        wall_edges, wall_velocity = pp._mask_image_wall(paths["RawVideos"], paths["Masked"], model_path,baffle_edges,dt, CF, skip_frame=skip_value, save_mask=True)
        
        # Call the PIV analysis function
        pp.piv_analysis(paths["Masked"], paths['Results'], skip_value, image_seq, cpus=n_cpus)
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
        #  Plot a video 
        if plot_video:
            results_file = Path(paths["Results"]).rglob("*.txt")
            pp.create_video_from_images(
                wall_edges, baffle_edges, results_file, paths['Results'],wall_velocity, v_max=v_max, sl=sl, Normalize_data=True,dt=dt, CF=CF  # # Set the variable to plot (v_mag, w_vor, or s33)
            )
    print("Your data is ready for analysis!")

def process_main_local():
    parser = argparse.ArgumentParser(description="Process PIV data.")
    parser.add_argument("--base_filepath", type=str, help="Base file path for raw videos.")
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    parser.add_argument("--normalize_data", type=bool, default=True, help="Normalize data flag.")
    parser.add_argument("--n_cpus", type=int, default=4, help="Number of CPUs to use.")
    parser.add_argument("--plot_video", type=bool, default=True, help="Flag to plot video.")
    parser.add_argument("--metadata_file", type=str, default=None, help="Path to metadata file.")

    args = parser.parse_args()

    main_processor(args.base_filepath, args.model_path, args.normalize_data, args.n_cpus, args.plot_video,  args.metadata_file)

if __name__ == "__main__":
    process_main_local()