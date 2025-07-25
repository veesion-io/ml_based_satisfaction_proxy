#!/usr/bin/env python3
"""
Preprocesses camera data to compute and save ground truth densities.

This script calculates the TP/FP probability densities for each camera using
a Kernel Density Estimate (KDE) with cross-validated bandwidth optimization.
It runs the calculations in parallel and saves the results to a single file
to be used by the training script.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import List, Tuple, Dict, Optional
import warnings
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed
import os
import random

warnings.filterwarnings('ignore')

# --- Configuration ---
N_BINS = 20 # Must match the training script configuration
TP_BANDWIDTH = 0.0546 # Globally determined optimal value
FP_BANDWIDTH = 0.0336 # Globally determined optimal value

def get_ground_truth_kde(data: np.ndarray, n_bins: int, bandwidth: float) -> Optional[np.ndarray]:
    """Calculates a smooth probability density using KDE with a fixed bandwidth."""
    if len(data) < 1:
        return np.zeros(n_bins)
        
    try:
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(data[:, None])
        
        x_eval = np.linspace(0, 1, n_bins)[:, None]
        log_density = kde.score_samples(x_eval)
        density = np.exp(log_density)
        
        return density / (density.sum() + 1e-6)
    except Exception as e:
        print(f"Warning: KDE calculation failed. Error: {e}")
        return None

def process_camera_file(file_path: str, n_bins: int) -> Optional[Dict]:
    """
    Reads a camera file, filters it, and gets the KDE density for TP/FP.
    """
    df = pd.read_parquet(file_path)
    
    tp_count = df['is_theft'].sum()
    fp_count = len(df) - tp_count

    if len(df) < 300 or tp_count < 5 or fp_count < 5:  # Need min 5 samples for stable KDE
        return None
    
    tp_probs = df[df['is_theft'] == 1]['max_proba'].values
    fp_probs = df[df['is_theft'] == 0]['max_proba'].values

    # Use KDE for smooth densities
    tp_density = get_ground_truth_kde(tp_probs, n_bins, TP_BANDWIDTH)
    fp_density = get_ground_truth_kde(fp_probs, n_bins, FP_BANDWIDTH)
    
    if tp_density is None or fp_density is None:
        return None
    
    return {
        'file_path': file_path,
        'tp_density': tp_density,
        'fp_density': fp_density
    }

def main():
    """Main function to run the parallel preprocessing."""
    data_dir = "data_by_camera"
    output_file = "ground_truth_histograms.pkl"
    
    print(f"Searching for camera files in {data_dir}...")
    all_camera_files = [str(f) for f in Path(data_dir).glob("*.parquet")]
    print(f"Found {len(all_camera_files)} camera files.")
    
    print("\nStarting parallel processing of all camera data with fixed bandwidths...")
    results = Parallel(n_jobs=-1)(
        delayed(process_camera_file)(f, N_BINS) for f in tqdm(all_camera_files)
    )
    
    # Filter out None results from cameras that were skipped
    valid_results = [r for r in results if r is not None]
    
    print(f"\nSuccessfully processed {len(valid_results)} cameras.")
    print(f"Saving processed densities to {output_file}...")
    
    with open(output_file, 'wb') as f:
        pickle.dump(valid_results, f)
        
    print("Preprocessing complete.")

if __name__ == "__main__":
    main() 