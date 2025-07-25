#!/usr/bin/env python3
"""
Load and test the asymmetric ratio model
"""

import torch
import numpy as np
import pandas as pd
from asymmetric_ratio_training import DeepSetsAsymmetricRatio
from density_prediction_training_with_ratio import ResidualMLP, MAB
import pickle

def load_asymmetric_model():
    """Load the trained asymmetric ratio model"""
    checkpoint = torch.load("runs/best_model/best_checkpoint_asymmetric.pth", map_location='cpu')
    
    # Get hyperparameters
    hyperparams = checkpoint['hyperparameters']
    phi_dim = hyperparams['phi_dim']
    num_heads = hyperparams['num_heads']
    n_bins = 20  # From training script
    
    # Create and load model
    model = DeepSetsAsymmetricRatio(phi_dim, n_bins, num_heads)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded asymmetric model with R² = {checkpoint['r2_score']:.4f}")
    print(f"Hyperparameters: {hyperparams}")
    
    return model

def predict_densities_and_ratio_asymmetric(model, alerts_df, k):
    """
    Predict TP/FP densities and ratio using the asymmetric model
    
    Args:
        model: Trained asymmetric ratio model
        alerts_df: DataFrame with 'max_proba' and 'is_theft' columns
        k: Sample size to use
    
    Returns:
        (tp_density, fp_density, tp_ratio): Predictions as numpy arrays
    """
    device = next(model.parameters()).device
    
    # Sample k alerts
    if len(alerts_df) < k:
        sampled_alerts = alerts_df.copy()
    else:
        sampled_alerts = alerts_df.sample(n=k, random_state=42)
    
    # Prepare features: [prob, is_theft, normalized_k]
    probs = sampled_alerts['max_proba'].values
    is_theft = sampled_alerts['is_theft'].astype(int).values
    k_normalized = np.log(len(sampled_alerts)) / np.log(2000)  # Same normalization as training
    
    # Create feature matrix
    features = np.column_stack([
        probs,
        is_theft, 
        np.full(len(sampled_alerts), k_normalized)
    ])
    
    # Pad to max sequence length and convert to tensor
    max_len = 2000  # From training
    if len(features) < max_len:
        padding = np.zeros((max_len - len(features), 3))
        features_padded = np.vstack([features, padding])
    else:
        features_padded = features[:max_len]
    
    features_tensor = torch.FloatTensor(features_padded).unsqueeze(0).to(device)
    counts_tensor = torch.LongTensor([len(sampled_alerts)]).to(device)
    
    # Predict
    with torch.no_grad():
        tp_pred, fp_pred, ratio_pred = model(features_tensor, counts_tensor)
        
        tp_density = tp_pred.squeeze(0).cpu().numpy()
        fp_density = fp_pred.squeeze(0).cpu().numpy()
        tp_ratio = ratio_pred.squeeze(0).cpu().numpy()
    
    return tp_density, fp_density, tp_ratio

def calculate_precision_from_asymmetric_predictions(tp_density, fp_density, tp_ratio, n_bins=20):
    """
    Calculate precision curve from asymmetric model predictions
    """
    # Bin edges for probabilities
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate cumulative counts from right to left (high prob to low prob)
    tp_counts = tp_density * tp_ratio  # Scale by ratio to get absolute counts
    fp_counts = fp_density * (1 - tp_ratio)
    
    # Cumulative sums from right to left (thresholds from high to low)
    tp_cumsum = np.cumsum(tp_counts[::-1])[::-1]
    fp_cumsum = np.cumsum(fp_counts[::-1])[::-1]
    
    # Calculate precision at each threshold
    total_positive = tp_cumsum + fp_cumsum
    precision = np.where(total_positive > 0, tp_cumsum / total_positive, 0)
    
    # Calculate average precision using trapezoidal rule
    # Weight by the recall (TP rate) to get proper AP
    total_tp = tp_counts.sum()
    if total_tp > 0:
        recall = tp_cumsum / total_tp
        # Use trapezoidal integration
        avg_precision = np.trapz(precision, recall)
    else:
        avg_precision = 0.0
    
    return avg_precision, precision, bin_centers

if __name__ == "__main__":
    # Load model
    model = load_asymmetric_model()
    
    # Test on a sample camera
    with open("ground_truth_histograms.pkl", 'rb') as f:
        data = pickle.load(f)
    
    # Pick a test camera
    test_camera = data[100]
    alerts_df = pd.read_parquet(test_camera['file_path'])
    
    print(f"\nTesting on camera: {test_camera['file_path']}")
    print(f"Total alerts: {len(alerts_df)}")
    
    # Test with different sample sizes
    for k in [50, 200, 500]:
        if len(alerts_df) >= k:
            tp_density, fp_density, tp_ratio = predict_densities_and_ratio_asymmetric(model, alerts_df, k)
            avg_precision, _, _ = calculate_precision_from_asymmetric_predictions(tp_density, fp_density, tp_ratio)
            
            print(f"k={k}: Predicted TP ratio = {tp_ratio:.3f}, Avg Precision = {avg_precision:.3f}") 