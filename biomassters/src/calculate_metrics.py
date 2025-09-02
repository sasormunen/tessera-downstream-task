import os
import numpy as np
import rasterio
#from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from glob import glob
#import seaborn as sns
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def calculate_average_rmse(pred_dir, truth_dir, out_csv= "per_chip_metrics.csv", tif=False):

    """ Function for calculating rmse and other metrics from predictions 
        For speed on plotting, the results are already calculated in folder "metrics" """
    
    pred_dir = Path(pred_dir)
    truth_dir = Path(truth_dir)

    chip_rmses = []
    records = []

    if tif == False:
        ending = ".npy"
    else:
        ending = ".tif"

    total_bias = 0
    n_pixels = 0
    ss_res = 0
    ss_tot = 0
    n_truth_pixels = 0

    mean_truth = 0
    for agbfile in truth_dir.glob("*_agbm"+ending):

        if tif:
            with rasterio.open(agbfile) as src:
                truth = src.read(1)
        else:
            truth = np.load(agbfile)

        #mask = ~np.isnan(truth)
        mean_truth += np.sum(truth)
        n_truth_pixels += truth.size
            
    mean_truth = mean_truth/n_truth_pixels
    
    for pred_path in pred_dir.glob("*_agbm"+ending):
        chip_id = pred_path.stem.replace("_agbm", "")
        truth_path = truth_dir / f"{chip_id}_agbm{ending}"
        #truth_path = truth_dir / f"{chip_id}_agbm"+ending
        if not truth_path.exists():
            print(f"Missing ground truth for {chip_id}")
            continue

        if tif:
            with rasterio.open(pred_path) as src:
                pred = src.read(1)
            with rasterio.open(truth_path) as src:
                truth = src.read(1)
        else:
            pred = np.load(pred_path)
            truth = np.load(truth_path)

        if pred.shape != truth.shape:
            print(f"Shape mismatch for {chip_id}")
            continue

        #mask = ~np.isnan(pred) & ~np.isnan(truth)
        pred_valid = pred #[mask]
        truth_valid = truth #[mask]
        if np.isnan(pred).any():
            print("nan in pred")
        if np.isnan(truth).any():
            print("nan in truth")

        if len(pred_valid) == 0:
            print(f"No valid pixels for {chip_id}")
            continue

        mse = np.mean((pred_valid - truth_valid) ** 2)
        rmse = np.sqrt(mse)
        chip_rmses.append(rmse)

        total_bias += np.sum(pred_valid-truth_valid)
        n_pixels += pred_valid.size #len(pred_valid)

        ss_res += np.sum((truth_valid-pred_valid)**2)
        ss_tot += np.sum((truth_valid - mean_truth)**2)
        
        #records.append({"chip_id": chip_id, "RMSE": rmse})

    # Compute average RMSE across chips
    if n_pixels != len(chip_rmses)*256*256 or n_truth_pixels != len(chip_rmses)*256*256:
        print("problem in chip count", len(chip_rmses), n_truth_pixels)
    avg_rmse = np.mean(chip_rmses)
    mean_bias = total_bias/n_pixels
    r2 = 1- ss_res/ ss_tot
    print(f"Average per-chip RMSE: {avg_rmse:.4f}")
    print(avg_rmse, ",", r2,",", mean_bias)
    
    return avg_rmse, r2, mean_bias

    
# Example usage:
#pred_dir = None #path to predictions
#gt_dir = None #ground truth dir
#tif = True if the predictions are .tif, False if .npy
#rmse, r2, mb = calculate_average_rmse(pred_dir, gt_dir, tif=False)