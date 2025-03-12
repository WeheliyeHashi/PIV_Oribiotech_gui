# PIV Processor GUI

## Overview
The PIV Processor GUI is a Python application designed to facilitate the processing of Particle Image Velocimetry (PIV) data. This application provides a user-friendly graphical interface that allows users to easily configure parameters for processing PIV data, including selecting directories for raw videos and model files, specifying the number of CPUs to use, and choosing options for data normalization and video plotting.

## Features
- Select base file path for raw videos.
- Select model path directory.
- Input for the number of CPUs to utilize during processing.
- Checkboxes for options to normalize data and save images.
- Option to include a metadata file directory or it will locate the metadata in the Auxiliary Folder. An error will return if there isnt any metadata file. 

## Requirements
To run this project, you need to have the following Python packages installed:
- pandas
- matplotlib
- sleap
- numpy
- tables
- skimage
- Tkinter 

You can install the required packages using the following command:
```
conda env create -f requirements.yml
conda activate PIV_code_ORI
pip install -e .
```

## Usage
1. Clone the repository or download the source code.
2. Navigate to the project directory.
3. Install the required dependencies.
4. Run the GUI application by executing the following command:
   ```
   conda activate PIV_code_ORI
   gui_piv
   ```
5. Use the interface to select the necessary files and options, then start the processing.


## Model

The model for detecting the walls can be found in sharepoint: Bioprocessing/Documents/OBP-EMC- Engineering Mixing Characterisation/PIV/EMC-0061



## Contributing
Contributions to improve the functionality and usability of the PIV Processor GUI are welcome. Please feel free to submit issues or pull requests.


