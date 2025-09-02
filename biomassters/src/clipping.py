import os
import numpy as np
from glob import glob
import rasterio

input_dir = "path"
output_dir = "path_clipped"

os.makedirs(output_dir, exist_ok=True)

threshold = 500
npy_files = glob(os.path.join(input_dir, "*.npy"))

for path in npy_files:
    
    arr = np.load(path)
    arr = np.clip(arr, a_min=None, a_max=threshold)
    out_path = os.path.join(output_dir, os.path.basename(path))
    np.save(out_path, arr)