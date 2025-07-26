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
from analysis.data_utils import load_and_filter_camera_data

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

def process_camera_file(camera_info: Dict, n_bins: int) -> Optional[Dict]:
    """
    Reads a camera file, filters it, and gets the KDE density for TP/FP.
    """
    df = camera_info['df']
    tp_probs = df[df['is_theft'] == 1]['max_proba'].values
    fp_probs = df[df['is_theft'] == 0]['max_proba'].values

    # Use KDE for smooth densities
    tp_density = get_ground_truth_kde(tp_probs, n_bins, TP_BANDWIDTH)
    fp_density = get_ground_truth_kde(fp_probs, n_bins, FP_BANDWIDTH)
    
    if tp_density is None or fp_density is None:
        return None
    
    # Calculate ground truth precision
    tp_count = camera_info['tp_count']
    fp_count = camera_info['fp_count']
    total_count = tp_count + fp_count
    tp_ratio_gt = tp_count / total_count if total_count > 0 else 0.0

    return {
        'file_path': camera_info['file_path'],
        'tp_density': tp_density,
        'fp_density': fp_density,
        'tp_count': camera_info['tp_count'],
        'fp_count': camera_info['fp_count'],
        'tp_ratio_gt': tp_ratio_gt
    }

def main():
    """Main function to run the parallel preprocessing."""
    output_file = "ground_truth_histograms.pkl"
    
    # Load and filter camera data using the new utility
    valid_cameras = load_and_filter_camera_data()
    
    print("\nStarting parallel processing of all camera data with fixed bandwidths...")
    results = Parallel(n_jobs=-1)(
        delayed(process_camera_file)(cam_info, N_BINS) for cam_info in tqdm(valid_cameras)
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