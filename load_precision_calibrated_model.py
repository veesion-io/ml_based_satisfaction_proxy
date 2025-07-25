#!/usr/bin/env python3
"""
Load and test the precision-calibrated model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from precision_calibrated_training import DeepSetsPrecisionCalibrated, compute_ground_truth_precision_per_bin

def load_precision_calibrated_model():
    """Load the trained precision-calibrated model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint_path = "runs/best_model/best_checkpoint_precision_calibrated.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get hyperparameters
    hyperparams = checkpoint['hyperparameters']
    phi_dim = hyperparams['phi_dim']
    num_heads = hyperparams['num_heads']
    
    # Create model
    model = DeepSetsPrecisionCalibrated(phi_dim, 20, num_heads).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded precision-calibrated model")
    print(f"   R² scores: Overall={checkpoint['r2_score']:.4f}, TP={checkpoint['r2_tp']:.4f}, FP={checkpoint['r2_fp']:.4f}")
    print(f"   Ratio R²={checkpoint['r2_ratio']:.4f}, Precision R²={checkpoint['r2_precision']:.4f}")
    
    return model

def predict_densities_ratio_and_precision(model, alerts_df, k):
    """Predict TP/FP densities, ratio, and calibrated precision using the precision-calibrated model"""
    device = next(model.parameters()).device
    
    # Sample subset
    if k > len(alerts_df):
        k = len(alerts_df)
    sampled = alerts_df.sample(n=k, replace=False)
    
    # Prepare features: [prob, is_theft, normalized_k]
    probs = sampled['max_proba'].values
    is_theft = sampled['is_theft'].values
    k_normalized = np.log(k) / np.log(2000)
    
    # Create feature matrix and pad to max length
    max_len = 2000
    features = np.zeros((max_len, 3))
    features[:k, 0] = probs
    features[:k, 1] = is_theft
    features[:k, 2] = k_normalized
    
    # Convert to tensors
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    counts_tensor = torch.tensor([k], dtype=torch.long).to(device)
    
    # Predict
    with torch.no_grad():
        tp_density, fp_density, tp_ratio, calibrated_precision = model(features_tensor, counts_tensor)
    
    return (tp_density.cpu().numpy()[0], 
            fp_density.cpu().numpy()[0], 
            tp_ratio.cpu().item(),
            calibrated_precision.cpu().numpy()[0])

def predict_average_precision_calibrated_model(model, alerts_df, k):
    """Predict average precision using the precision-calibrated model (matching original script method)"""
    
    # Sample subset (like original script)
    if k > len(alerts_df):
        k = len(alerts_df)
    sampled = alerts_df.sample(n=k, replace=False)
    
    # Get predictions from precision-calibrated model
    tp_density, fp_density, tp_ratio, calibrated_precision = predict_densities_ratio_and_precision(model, alerts_df, k)
    
    # Calculate predicted counts for each bin (matching original script)
    total_predicted_count = k
    fp_ratio = 1.0 - tp_ratio
    
    # Scale densities to get predicted counts per bin
    tp_counts = tp_density * tp_ratio * total_predicted_count
    fp_counts = fp_density * fp_ratio * total_predicted_count
    total_counts_per_bin = tp_counts + fp_counts
    
    # Use the calibrated precision directly (this is what we trained for!)
    precision_per_bin = calibrated_precision
    
    # Calculate weighted average precision (same as original script)
    weights = total_counts_per_bin
    total_weight = np.sum(weights)
    
    if total_weight < 1e-9:
        return 0.0  # No valid predictions
    
    average_precision = np.sum(precision_per_bin * weights) / total_weight
    
    return average_precision

def compute_ground_truth_average_precision(alerts_df):
    """Compute ground truth average precision from full camera data using binning method (same as original script)"""
    # Use the same method as the original script
    bins = np.linspace(0, 1, 21)  # 20 bins
    
    precision_values = []
    weights = []  # Number of alerts in each bin
    
    for i in range(len(bins) - 1):
        # Get alerts in this probability bin
        mask = (alerts_df['max_proba'] >= bins[i]) & (alerts_df['max_proba'] < bins[i+1])
        bin_data = alerts_df[mask]
        
        if len(bin_data) == 0:
            continue  # Skip empty bins
        
        tp_count = len(bin_data[bin_data['is_theft'] == 1])
        total_count = len(bin_data)
        precision = tp_count / total_count
        
        precision_values.append(precision)
        weights.append(total_count)
    
    if not precision_values:
        return 0.0  # No valid bins
    
    # Calculate weighted average precision
    precision_values = np.array(precision_values)
    weights = np.array(weights)
    average_precision = np.average(precision_values, weights=weights)
    
    return average_precision

if __name__ == "__main__":
    print("🧪 Testing Precision-Calibrated Model")
    print("=" * 40)
    
    # Load model
    model = load_precision_calibrated_model()
    
    # Test on a sample camera
    import pickle
    with open("ground_truth_histograms.pkl", 'rb') as f:
        data = pickle.load(f)
    
    # Test camera
    test_camera = data[100]
    alerts_df = pd.read_parquet(test_camera['file_path'])
    
    print(f"\n📊 Testing on camera with {len(alerts_df)} alerts")
    
    # Test different sample sizes
    sample_sizes = [50, 100, 200, 500, 1000]
    
    print(f"\n{'Sample Size':<12} {'Predicted AP':<14} {'Ground Truth AP':<16} {'Error':<10}")
    print("-" * 55)
    
    gt_ap = compute_ground_truth_average_precision(alerts_df)
    
    for k in sample_sizes:
        if k <= len(alerts_df):
            pred_ap = predict_average_precision_calibrated_model(model, alerts_df, k)
            error = abs(pred_ap - gt_ap)
            
            print(f"{k:<12} {pred_ap:<14.4f} {gt_ap:<16.4f} {error:<10.4f}")
    
    print(f"\n🎯 Ground Truth AP: {gt_ap:.4f}")
    print("✅ Model test complete!") 