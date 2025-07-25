#!/usr/bin/env python3
"""
Load Best Model with TP Ratio Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_BINS = 20

class ResidualMLP(nn.Module):
    def __init__(self, dim): 
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
    
    def forward(self, x): 
        return F.relu(self.fc2(F.relu(self.fc1(x))) + x)

class MAB(nn.Module):
    def __init__(self, dim_V, num_heads, ln=True):
        super(MAB, self).__init__()
        self.mha = nn.MultiheadAttention(dim_V, num_heads)
        self.ln1 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.ln2 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.ffn = nn.Sequential(nn.Linear(dim_V, dim_V), nn.ReLU(), nn.Linear(dim_V, dim_V))
    
    def forward(self, Q, K):
        Q_norm, K_norm = self.ln1(Q).permute(1, 0, 2), self.ln1(K).permute(1, 0, 2)
        out, _ = self.mha(Q_norm, K_norm, K_norm)
        out = Q + out.permute(1, 0, 2)
        out = self.ln2(out)
        out = out + self.ffn(out)
        return out

class DeepSetsAdvancedWithRatio(nn.Module):
    def __init__(self, phi_dim, n_bins, num_heads):
        super(DeepSetsAdvancedWithRatio, self).__init__()
        if phi_dim % num_heads != 0: 
            phi_dim = (phi_dim // num_heads + 1) * num_heads
        
        self.phi = nn.Sequential(
            nn.Linear(3, phi_dim), 
            ResidualMLP(phi_dim), 
            ResidualMLP(phi_dim)
        )
        self.pooling = MAB(phi_dim, num_heads)
        self.query = nn.Parameter(torch.randn(1, 1, phi_dim))
        self.rho = nn.Sequential(ResidualMLP(phi_dim), ResidualMLP(phi_dim))
        
        # Three output heads: TP density, FP density, and TP ratio
        self.tp_head = nn.Linear(phi_dim, n_bins)
        self.fp_head = nn.Linear(phi_dim, n_bins)
        self.ratio_head = nn.Linear(phi_dim, 1)  # Single output for TP ratio
    
    def forward(self, x, counts):
        mask = torch.arange(x.size(1), device=x.device)[None, :] < counts[:, None]
        phi_out = self.phi(x * mask.unsqueeze(-1))
        query = self.query.repeat(x.shape[0], 1, 1)
        agg = self.pooling(query, phi_out).squeeze(1)
        rho_out = self.rho(agg)
        
        tp_out = F.softmax(self.tp_head(rho_out), dim=1)
        fp_out = F.softmax(self.fp_head(rho_out), dim=1)
        ratio_out = torch.sigmoid(self.ratio_head(rho_out)).squeeze(-1)  # Sigmoid for [0,1] range
        
        return tp_out, fp_out, ratio_out

def load_best_model_with_ratio():
    """Load the best model with TP ratio prediction"""
    
    checkpoint_path = "runs/best_model/best_checkpoint_with_ratio.pth"
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Get hyperparameters
    hyperparams = checkpoint['hyperparameters']
    
    # Create model with same architecture
    model = DeepSetsAdvancedWithRatio(
        phi_dim=hyperparams['phi_dim'],
        n_bins=N_BINS,
        num_heads=hyperparams['num_heads']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print("✅ Model with TP ratio loaded successfully!")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   R² Score: {checkpoint['r2_score']:.4f}")
    print(f"   TP R²: {checkpoint['r2_tp']:.4f}")
    print(f"   FP R²: {checkpoint['r2_fp']:.4f}")
    print(f"   Ratio R²: {checkpoint['r2_ratio']:.4f}")
    print(f"   Hyperparameters: {hyperparams}")
    
    return model, checkpoint

def predict_densities_and_ratio(model, sample_data, sample_size=None):
    """
    Predict TP/FP densities and TP ratio from sample data
    
    Args:
        model: Trained DeepSetsAdvancedWithRatio model
        sample_data: DataFrame with 'max_proba' and 'is_theft' columns
        sample_size: Optional override for sample size (uses len(sample_data) if None)
    
    Returns:
        tuple: (tp_density, fp_density, tp_ratio) as numpy arrays
    """
    
    if sample_size is None:
        sample_size = len(sample_data)
    
    # Prepare features
    base_features = torch.tensor(
        sample_data[['max_proba', 'is_theft']].values, 
        dtype=torch.float32
    )
    
    # Add sample size feature (log-normalized)
    max_size = 2000  # From training range
    k_normalized = np.log(sample_size) / np.log(max_size)
    k_feature = torch.full((len(sample_data), 1), fill_value=k_normalized, dtype=torch.float32)
    
    # Combine features
    features = torch.cat([base_features, k_feature], dim=1)
    features = features.unsqueeze(0).to(DEVICE)  # Add batch dimension
    counts = torch.tensor([len(sample_data)], dtype=torch.float32).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        tp_density, fp_density, tp_ratio = model(features, counts)
    
    return (tp_density.squeeze().cpu().numpy(), 
            fp_density.squeeze().cpu().numpy(),
            tp_ratio.squeeze().cpu().numpy())

def calculate_precision_from_predictions(tp_density, fp_density, tp_ratio):
    """
    Calculate precision values from predicted densities and ratio
    
    Args:
        tp_density: TP probability density (N_BINS,)
        fp_density: FP probability density (N_BINS,)
        tp_ratio: TP ratio (scalar)
    
    Returns:
        numpy array: Precision values for each probability bin
    """
    
    # Scale densities by the ratio to get actual counts
    # tp_ratio = TP_count / Total_count
    # fp_ratio = FP_count / Total_count = (1 - tp_ratio)
    fp_ratio = 1.0 - tp_ratio
    
    # Scale the densities
    tp_scaled = tp_density * tp_ratio
    fp_scaled = fp_density * fp_ratio
    
    # Calculate precision = TP / (TP + FP) for each bin
    precision = tp_scaled / (tp_scaled + fp_scaled + 1e-9)
    
    return precision

def predict_average_precision_ratio_model(model, alerts_df, k):
    """Predict average precision using the ratio model"""
    # Sample subset
    if k > len(alerts_df):
        k = len(alerts_df)
    sampled = alerts_df.sample(n=k, replace=False)
    
    # Get predictions
    tp_density, fp_density, tp_ratio = predict_densities_and_ratio(model, sampled, k)
    
    # Calculate precision curve
    precision = calculate_precision_from_predictions(tp_density, fp_density, tp_ratio)
    
    # Weight by the total alert distribution
    fp_ratio = 1.0 - tp_ratio
    total_density = tp_density * tp_ratio + fp_density * fp_ratio
    total_density = total_density / (total_density.sum() + 1e-9)  # Normalize
    
    # Calculate weighted average precision
    weighted_precision = np.sum(precision * total_density)
    
    return weighted_precision

if __name__ == "__main__":
    # Test the model loading
    model, info = load_best_model_with_ratio()
    
    # Test prediction on a sample
    import pickle
    with open('ground_truth_histograms.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Use first camera as test
    test_camera = data[0]
    test_df = pd.read_parquet(test_camera['file_path'])
    sample_df = test_df.sample(n=100, replace=False)
    
    print(f"\nTesting prediction on sample of {len(sample_df)} alerts...")
    tp_density, fp_density, tp_ratio = predict_densities_and_ratio(model, sample_df)
    
    print(f"Predicted TP ratio: {tp_ratio:.3f}")
    print(f"Actual TP ratio: {len(sample_df[sample_df['is_theft'] == 1]) / len(sample_df):.3f}")
    
    # Calculate precision curve
    precision = calculate_precision_from_predictions(tp_density, fp_density, tp_ratio)
    print(f"Precision range: {precision.min():.3f} - {precision.max():.3f}")
    
    print("\n🎉 Model test successful!") 