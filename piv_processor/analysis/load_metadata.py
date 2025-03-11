import pandas as pd
import numpy as np


def _return_metadata(metadata_row):
    v_max = metadata_row.iloc[5]
    sl = metadata_row.iloc[3]
    wall_bottom_left = metadata_row.iloc[[10, 12]].astype(int).values
    wall_bottom_right = metadata_row.iloc[[11, 12]].astype(int).values
    skip_value = int(metadata_row.iloc[8])
    image_seq = metadata_row.iloc[9].strip('"')
    dt = int(metadata_row.iloc[7])  # ms time interval between frames

    return wall_bottom_left, wall_bottom_right, skip_value, image_seq, dt, v_max, sl
