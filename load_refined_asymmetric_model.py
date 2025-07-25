#!/usr/bin/env python3
"""
Load and test the refined asymmetric ratio model
"""

import torch
import numpy as np
import pandas as pd
from asymmetric_ratio_training_v2 import DeepSetsRefinedAsymmetric
import pickle

def load_refined_asymmetric_model():
    """Load the trained refined asymmetric ratio model"""
    checkpoint = torch.load("runs/best_model/best_checkpoint_refined_asymmetric.pth", map_location='cpu')
    
    # Get hyperparameters
    hyperparams = checkpoint['hyperparameters']
    phi_dim = hyperparams['phi_dim']
    num_heads = hyperparams['num_heads']
    n_bins = 20  # From training script
    
    # Create and load model
    model = DeepSetsRefinedAsymmetric(phi_dim, n_bins, num_heads)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded refined asymmetric model with R² = {checkpoint['r2_score']:.4f}")
    print(f"Hyperparameters: {hyperparams}")
    
    return model

def predict_densities_and_ratio_refined(model, alerts_df, k):
    """
    Predict TP/FP densities and ratio using the refined asymmetric model
    
    Args:
        model: Trained refined asymmetric ratio model
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

if __name__ == "__main__":
    # Load model
    model = load_refined_asymmetric_model()
    
    # Test on a sample camera
    with open("ground_truth_histograms.pkl", 'rb') as f:
        data = pickle.load(f)
    
    # Pick a test camera
    test_camera = data[100]
    alerts_df = pd.read_parquet(test_camera['file_path'])
    
    print(f"\nTesting on camera: {test_camera['file_path']}")
    print(f"Total alerts: {len(alerts_df)}")
    
    # Ground truth
    actual_tp = len(alerts_df[alerts_df['is_theft'] == 1])
    actual_ratio = actual_tp / len(alerts_df)
    print(f"Ground truth TP ratio: {actual_ratio:.4f}")
    
    # Test with different sample sizes
    for k in [50, 200, 500]:
        if len(alerts_df) >= k:
            tp_density, fp_density, tp_ratio = predict_densities_and_ratio_refined(model, alerts_df, k)
            print(f"k={k}: Predicted TP ratio = {tp_ratio:.4f} (vs GT {actual_ratio:.4f})") 