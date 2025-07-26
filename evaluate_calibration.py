#!/usr/bin/env python3
"""
Evaluates the calibration of the model's predicted precision distributions.
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

from analysis.model_loader import load_precision_aware_model, predict_densities_and_ratio_precision_aware
from models.training import load_data
from analysis.distribution_utils import mixture_logistic_cdf_numpy

def get_mixture_quantile(weights, locations, scales, quantile):
    """
    Numerically finds the quantile of a mixture of logistic distributions.
    """
    # Search for the value x such that CDF(x) = quantile
    low = 0.0
    high = 1.0
    for _ in range(100):  # 100 iterations of binary search are sufficient for high precision
        mid = (low + high) / 2
        cdf_val = mixture_logistic_cdf_numpy(np.array([mid]), weights, locations, scales)[0]
        if cdf_val < quantile:
            low = mid
        else:
            high = mid
    return (low + high) / 2

def evaluate_calibration(data_fraction=1.0):
    """
    Calculates the percentage of time the ground truth falls within the predicted 95% confidence interval.
    """
    print("Loading model and validation data...")
    model = load_precision_aware_model()
    _, val_data = load_data()
    print(f"Loaded {len(val_data)} cameras for calibration evaluation.")

    coverage_count = 0
    
    print(f"Evaluating calibration using {data_fraction:.0%} of data for each camera...")
    for camera_info in tqdm(val_data, desc="Evaluating Calibration"):
        # Load data and get ground truth
        camera_df = pd.read_parquet(camera_info['file_path'])
        gt_precision = camera_info['tp_ratio_gt']

        # Sample a fraction of the data for prediction
        sampled_df = camera_df.sample(frac=data_fraction, random_state=42) if data_fraction < 1.0 else camera_df
        sample_size = len(sampled_df)
        
        # Get model prediction
        _, _, weights, locations, scales = predict_densities_and_ratio_precision_aware(model, sampled_df, sample_size)
        
        # Calculate 95% confidence interval
        lower_bound = get_mixture_quantile(weights, locations, scales, 0.025)
        upper_bound = get_mixture_quantile(weights, locations, scales, 0.975)
        
        # Check if ground truth is within the interval
        if lower_bound <= gt_precision <= upper_bound:
            coverage_count += 1
            
    coverage_percentage = (coverage_count / len(val_data)) * 100
    print(f"\n✅ Calibration Evaluation Complete:")
    print(f"   The ground truth was contained within the 95% confidence interval for {coverage_percentage:.2f}% of the cameras.")
    print(f"   (A well-calibrated model should be close to 95%).")

if __name__ == "__main__":
    evaluate_calibration(data_fraction=1.0)
    evaluate_calibration(data_fraction=0.3) 