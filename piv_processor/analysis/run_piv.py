#from openpiv import windef  # <---- see windef.py for details
from piv_processor.PIV_packages.openpiv import windef
# from openpiv.settings import PIVSettings

import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt


plt.rcParams["image.cmap"] = "gray"

import cv2

# %%


def piv_analysis(filepath, filepath_results, skip_value, image_seq, cpus=1):
    """
    Performs PIV analysis with customizable settings.

    Parameters:
        filepath (str): Path to the folder containing the images.
        skip_value (int): Number of frames to skip for processing.
        image_seq (str): The sequence type, e.g., '(1+2),(2+3)' or '(1+2),(3+4)'.

    Returns:
        None: Outputs are saved to the specified directory.
    """
    # Step value based on the sequence
    step = 1 if image_seq == "(1+2),(2+3)" else 2

    # Initialize settings
    settings = windef.Settings()

    # Data related settings
    settings.filepath_images = filepath
    settings.save_path = Path(filepath_results)
    # print('Wall velocity analysis' if 'wall' in filepath else 'Fluid velocity analysis')
    # Supported file extensions
    valid_extensions = {".png", ".tif", ".jpeg", ".jpg"}

    # Get all files with valid extensions
    files = [
        file.name
        for file in Path(settings.filepath_images).iterdir()
        if file.suffix.lower() in valid_extensions
    ]

    # Check if no valid files are found
    if not files:
        raise ValueError(
            f"No valid image files found in '{settings.filepath_images}'. "
            f"The current code only supports the following extensions: {', '.join(valid_extensions)}"
        )
    settings.save_folder_suffix = "Exp_1"

    # Frame patterns based on sequence and skip value
    settings.frame_pattern_a = files[0::step]
    settings.frame_pattern_b = files[skip_value::step]

    # Processing Parameters
    settings.deformation_method = "symmetric"
    settings.correlation_method = "circular"
    settings.normalized_correlation = False
    settings.num_iterations = 3
    settings.windowsizes = (128, 64, 32)
    settings.overlap = (64, 32, 16)
    settings.subpixel_method = "gaussian"
    settings.interpolation_order = 3
    settings.scaling_factor = 1
    settings.dt = 1
    settings.n_cpus = cpus

    # Signal to noise ratio options
    settings.sig2noise_method = "peak2peak"
    settings.sig2noise_mask = 2

    # Validation Parameters
    settings.validation_first_pass = True
    settings.std_threshold = 7
    settings.median_threshold = 3
    settings.median_size = 1
    settings.sig2noise_threshold = 1.2

    # Outlier replacement or smoothing options
    settings.replace_vectors = True
    settings.smoothn = True
    settings.smoothn_p = 0.5
    settings.filter_method = "localmean"
    settings.max_filter_iteration = 4
    settings.filter_kernel_size = 2

    # Output options
    settings.save_plot = False
    settings.show_plot = False
    settings.scale_plot = 100
    windef.piv(settings)
    print(f"PIV analysis completed. Results saved in {settings.save_path}")
