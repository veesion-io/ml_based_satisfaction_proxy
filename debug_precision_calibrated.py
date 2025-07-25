#!/usr/bin/env python3
"""
Debug Precision-Calibrated Model
Compare the predictions between ratio model and precision-calibrated model
"""

import pickle
import numpy as np
import pandas as pd
from load_best_model_with_ratio import (
    load_best_model_with_ratio,
    predict_densities_and_ratio,
    calculate_precision_from_predictions
)
from load_precision_calibrated_model import (
    load_precision_calibrated_model,
    predict_densities_ratio_and_precision,
    compute_ground_truth_average_precision
)

def debug_models():
    """Compare what both models predict for the same camera"""
    
    print("🔍 DEBUGGING PRECISION-CALIBRATED vs RATIO MODEL")
    print("=" * 55)
    
    # Load both models
    ratio_model, _ = load_best_model_with_ratio()
    calibrated_model = load_precision_calibrated_model()
    
    # Load test data
    with open("ground_truth_histograms.pkl", 'rb') as f:
        data = pickle.load(f)
    
    # Test on a single camera
    test_camera = data[10]  # Pick a specific camera
    alerts_df = pd.read_parquet(test_camera['file_path'])
    
    # Limit to 2000 alerts for compatibility
    if len(alerts_df) > 2000:
        alerts_df = alerts_df.sample(n=2000, replace=False)
    
    print(f"📊 Testing camera with {len(alerts_df)} alerts")
    
    # Calculate ground truth
    gt_ap = compute_ground_truth_average_precision(alerts_df)
    print(f"🎯 Ground Truth Average Precision: {gt_ap:.4f}")
    
    # Test at different sample sizes
    sample_sizes = [100, 300, 500, 1000]
    
    print(f"\n{'Size':<6} {'GT AP':<8} {'Ratio AP':<10} {'Ratio Err':<10} {'Calib AP':<10} {'Calib Err':<10}")
    print("-" * 65)
    
    for k in sample_sizes:
        if k > len(alerts_df):
            k = len(alerts_df)
            
        # Sample the same subset for both models
        sampled = alerts_df.sample(n=k, replace=False, random_state=42)
        
        # === RATIO MODEL ===
        ratio_tp_density, ratio_fp_density, ratio_tp_ratio = predict_densities_and_ratio(ratio_model, sampled, k)
        ratio_precision = calculate_precision_from_predictions(ratio_tp_density, ratio_fp_density, ratio_tp_ratio)
        
        # Calculate ratio model AP
        ratio_fp_ratio = 1.0 - ratio_tp_ratio
        ratio_total_density = ratio_tp_density * ratio_tp_ratio + ratio_fp_density * ratio_fp_ratio
        ratio_total_density = ratio_total_density / (ratio_total_density.sum() + 1e-9)
        ratio_ap = np.sum(ratio_precision * ratio_total_density)
        
        # === PRECISION-CALIBRATED MODEL ===
        calib_tp_density, calib_fp_density, calib_tp_ratio, calib_precision = predict_densities_ratio_and_precision(calibrated_model, alerts_df, k)
        
        # Calculate calibrated model AP
        calib_fp_ratio = 1.0 - calib_tp_ratio
        calib_total_density = calib_tp_density * calib_tp_ratio + calib_fp_density * calib_fp_ratio
        calib_total_density = calib_total_density / (calib_total_density.sum() + 1e-9)
        calib_ap = np.sum(calib_precision * calib_total_density)
        
        # Calculate errors
        ratio_error = abs(ratio_ap - gt_ap)
        calib_error = abs(calib_ap - gt_ap)
        
        print(f"{k:<6} {gt_ap:<8.4f} {ratio_ap:<10.4f} {ratio_error:<10.4f} {calib_ap:<10.4f} {calib_error:<10.4f}")
    
    print(f"\n🔍 DETAILED COMPARISON (k=500):")
    print("=" * 40)
    
    k = 500
    sampled = alerts_df.sample(n=k, replace=False, random_state=42)
    
    # Get predictions from both models
    ratio_tp_density, ratio_fp_density, ratio_tp_ratio = predict_densities_and_ratio(ratio_model, sampled, k)
    ratio_precision = calculate_precision_from_predictions(ratio_tp_density, ratio_fp_density, ratio_tp_ratio)
    
    calib_tp_density, calib_fp_density, calib_tp_ratio, calib_precision = predict_densities_ratio_and_precision(calibrated_model, alerts_df, k)
    
    print(f"\n📊 TP Ratio:")
    print(f"   Ratio Model:      {ratio_tp_ratio:.4f}")
    print(f"   Calibrated Model: {calib_tp_ratio:.4f}")
    print(f"   Actual (sample):  {sampled['is_theft'].sum() / len(sampled):.4f}")
    print(f"   Actual (full):    {alerts_df['is_theft'].sum() / len(alerts_df):.4f}")
    
    print(f"\n📊 Precision Values (first 10 bins):")
    print(f"   Ratio Model:      {ratio_precision[:10]}")
    print(f"   Calibrated Model: {calib_precision[:10]}")
    
    print(f"\n📊 Density Comparison (first 10 bins):")
    print(f"   Ratio TP density:    {ratio_tp_density[:10]}")
    print(f"   Calib TP density:    {calib_tp_density[:10]}")
    print(f"   Ratio FP density:    {ratio_fp_density[:10]}")
    print(f"   Calib FP density:    {calib_fp_density[:10]}")
    
    # Check if the precision values make sense
    print(f"\n🔍 PRECISION ANALYSIS:")
    print(f"   Ratio precision range:  {ratio_precision.min():.4f} - {ratio_precision.max():.4f}")
    print(f"   Calib precision range:  {calib_precision.min():.4f} - {calib_precision.max():.4f}")
    print(f"   Ratio precision mean:   {ratio_precision.mean():.4f}")
    print(f"   Calib precision mean:   {calib_precision.mean():.4f}")
    
    # Check for obvious issues
    if np.any(calib_precision > 1.0):
        print("   ⚠️ WARNING: Calibrated precision > 1.0 detected!")
    if np.any(calib_precision < 0.0):
        print("   ⚠️ WARNING: Calibrated precision < 0.0 detected!")
    if np.any(np.isnan(calib_precision)):
        print("   ⚠️ WARNING: NaN in calibrated precision!")

if __name__ == "__main__":
    debug_models() 