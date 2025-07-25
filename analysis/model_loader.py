#!/usr/bin/env python3
"""
Model loading and prediction utilities for convergence analysis
"""

import numpy as np
import torch
import torch.nn.functional as F
from models.architecture import ResidualMLP, MAB

N_BINS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeepSetsPrecisionAware(torch.nn.Module):
    def __init__(self, phi_dim, n_bins, num_heads):
        super(DeepSetsPrecisionAware, self).__init__()
        if phi_dim % num_heads != 0: 
            phi_dim = (phi_dim // num_heads + 1) * num_heads
        
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(3, phi_dim), 
            ResidualMLP(phi_dim), 
            ResidualMLP(phi_dim)
        )
        self.pooling = MAB(phi_dim, num_heads)
        self.query = torch.nn.Parameter(torch.randn(1, 1, phi_dim))
        self.rho = torch.nn.Sequential(ResidualMLP(phi_dim), ResidualMLP(phi_dim))
        
        # Three output heads: TP density, FP density, and TP ratio mixture distribution
        self.tp_head = torch.nn.Linear(phi_dim, n_bins)
        self.fp_head = torch.nn.Linear(phi_dim, n_bins)
        
        # Mixture of logistic components for TP ratio distribution
        self.num_mixture_components = 5  # Number of logistic components
        self.mixture_weights_head = torch.nn.Linear(phi_dim, self.num_mixture_components)  # Mixture weights
        self.mixture_locations_head = torch.nn.Linear(phi_dim, self.num_mixture_components)  # Location parameters μᵢ
        self.mixture_scales_head = torch.nn.Linear(phi_dim, self.num_mixture_components)  # Scale parameters sᵢ
    
    def forward(self, x, counts):
        mask = torch.arange(x.size(1), device=x.device)[None, :] < counts[:, None]
        phi_out = self.phi(x * mask.unsqueeze(-1))
        query = self.query.repeat(x.shape[0], 1, 1)
        agg = self.pooling(query, phi_out).squeeze(1)
        rho_out = self.rho(agg)
        
        tp_out = F.softmax(self.tp_head(rho_out), dim=1)
        fp_out = F.softmax(self.fp_head(rho_out), dim=1)
        
        # Mixture of logistic components for TP ratio distribution
        mixture_weights_raw = self.mixture_weights_head(rho_out)  # (B, num_components)
        mixture_locations_raw = self.mixture_locations_head(rho_out)  # (B, num_components)
        mixture_scales_raw = self.mixture_scales_head(rho_out)  # (B, num_components)
        
        # Normalize mixture weights
        mixture_weights = F.softmax(mixture_weights_raw, dim=1)
        
        # Constrain locations to [0, 1] using sigmoid
        mixture_locations = torch.sigmoid(mixture_locations_raw)
        
        # Constrain scales to be positive using softplus
        mixture_scales = F.softplus(mixture_scales_raw) + 1e-6  # Add small constant for numerical stability
        
        return tp_out, fp_out, mixture_weights, mixture_locations, mixture_scales

def load_precision_aware_model():
    """Load the precision-aware model from checkpoint"""
    checkpoint = torch.load("runs/best_model/best_checkpoint_precision_aware.pth", map_location=DEVICE)
    hyperparams = checkpoint['hyperparameters']
    model = DeepSetsPrecisionAware(hyperparams['phi_dim'], N_BINS, hyperparams['num_heads']).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_densities_and_ratio_precision_aware(model, sample_data, sample_size):
    """Make predictions using the precision-aware model"""
    features = np.concatenate([
        sample_data[['max_proba', 'is_theft']].values,
        np.full((len(sample_data), 1), np.log(sample_size) / np.log(2000))
    ], axis=1)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    counts = torch.tensor([sample_size], dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        tp_density, fp_density, mixture_weights, mixture_locations, mixture_scales = model(features_tensor, counts)
    
    return (tp_density.cpu().numpy()[0], 
            fp_density.cpu().numpy()[0], 
            mixture_weights.cpu().numpy()[0],
            mixture_locations.cpu().numpy()[0], 
            mixture_scales.cpu().numpy()[0])

def calculate_simple_tp_ratio(camera_data):
    """Calculate simple TP ratio (same as what the model predicts)"""
    tp_count = len(camera_data[camera_data['is_theft'] == 1])
    total_count = len(camera_data)
    
    if total_count == 0:
        return 0.0
    
    return tp_count / total_count

def predict_average_precision_aware(model, sample_data, sample_size):
    """Predict TP ratio from sample data using the precision-aware model with uncertainty"""
    # Get predictions from precision-aware model (now returns mixture of logistic distributions)
    _, _, mixture_weights, mixture_locations, mixture_scales = predict_densities_and_ratio_precision_aware(
        model, sample_data, sample_size=sample_size
    )
    
    # Extract distribution information  
    from .distribution_utils import extract_tp_ratio_distribution_info
    dist_info = extract_tp_ratio_distribution_info(mixture_weights, mixture_locations, mixture_scales)
    
    # Return mean as point estimate (for compatibility with existing code)
    return dist_info['mean']

def predict_average_precision_aware_with_uncertainty(model, sample_data, sample_size):
    """Predict TP ratio with full uncertainty information"""
    # Get predictions from precision-aware model
    _, _, mixture_weights, mixture_locations, mixture_scales = predict_densities_and_ratio_precision_aware(
        model, sample_data, sample_size=sample_size
    )
    
    # Return full distribution information
    from .distribution_utils import extract_tp_ratio_distribution_info
    return extract_tp_ratio_distribution_info(mixture_weights, mixture_locations, mixture_scales) 