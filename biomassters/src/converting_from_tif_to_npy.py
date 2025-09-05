import os
import numpy as np
import rasterio
from glob import glob

input_dir = "/m/cs/scratch/networks/silja/forests/btfm_project/efm/data/representations_train_all"
output_dir = input_dir


tif_files = glob(os.path.join(input_dir, "*.tif"))

for tif_path in tif_files:
    with rasterio.open(tif_path) as src:
        array = src.read()  # shape: (bands, H, W)
        array = np.transpose(array, (1, 2, 0))  # → (H, W, bands)
    
    filename = os.path.splitext(os.path.basename(tif_path))[0] + ".npy"
    out_path = os.path.join(output_dir, filename)
    np.save(out_path, array)
    #print(f"Saved {filename} with shape {array.shape}")
