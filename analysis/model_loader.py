#!/usr/bin/env python3
"""
Model loading and prediction utilities for convergence analysis
"""

import numpy as np
import torch
import torch.nn.functional as F
from models.architecture import DeepSetsPrecisionAware

N_BINS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_precision_aware_model():
    """Load the precision-aware model from checkpoint"""
    checkpoint = torch.load("runs/best_model/best_checkpoint_precision_aware.pth", map_location=DEVICE)
    hyperparams = checkpoint['hyperparameters']
    model = DeepSetsPrecisionAware(
        hyperparams['phi_dim'], 
        N_BINS, 
        hyperparams['num_heads'],
        hyperparams.get('dropout_rate', 0.0)  # Use dropout_rate from checkpoint, default to 0.0
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model

def predict_densities_and_ratio_precision_aware(model, sample_data, sample_size):
    """Make predictions using the precision-aware model"""
    
    # Prepare features with normalized counts
    features = sample_data[['max_proba', 'is_theft']].values
    counts_normalized = np.log(sample_size) / np.log(2000)
    counts_expanded = np.full((features.shape[0], 1), counts_normalized)
    features_with_counts = np.concatenate([features, counts_expanded], axis=1)

    features_tensor = torch.tensor(features_with_counts, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    counts_tensor = torch.tensor([sample_size], dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        mixture_weights, mixture_locations, mixture_scales, _ = model(features_tensor, counts_tensor)
    
    return (mixture_weights.cpu().numpy()[0],
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
    mixture_weights, mixture_locations, mixture_scales = predict_densities_and_ratio_precision_aware(
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
    mixture_weights, mixture_locations, mixture_scales = predict_densities_and_ratio_precision_aware(
        model, sample_data, sample_size=sample_size
    )
    
    # Return full distribution information
    from .distribution_utils import extract_tp_ratio_distribution_info
    return extract_tp_ratio_distribution_info(mixture_weights, mixture_locations, mixture_scales) 