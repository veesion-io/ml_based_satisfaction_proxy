#!/usr/bin/env python3
"""
Loss functions for precision-aware training
"""

import torch
import torch.nn.functional as F

def mixture_logistic_cdf(x, weights, locations, scales):
    """
    Compute CDF of mixture of logistic distributions: P(X ≤ x) = Σᵢ wᵢ * σ((x - μᵢ) / sᵢ)
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)  # (batch_size, 1)
    
    # Compute (x - μᵢ) / sᵢ for all components
    standardized = (x - locations) / scales  # (batch_size, num_components)
    
    # Compute σ((x - μᵢ) / sᵢ) for all components
    logistic_cdfs = torch.sigmoid(standardized)  # (batch_size, num_components)
    
    # Weighted sum: Σᵢ wᵢ * σ((x - μᵢ) / sᵢ)
    mixture_cdf = torch.sum(weights * logistic_cdfs, dim=1)  # (batch_size,)
    
    return mixture_cdf

def mixture_logistic_pdf(x, weights, locations, scales):
    """
    Compute PDF of mixture of logistic distributions: p(x) = Σᵢ wᵢ * (1/sᵢ) * σ((x - μᵢ) / sᵢ) * (1 - σ((x - μᵢ) / sᵢ))
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)  # (batch_size, 1)
    
    # Add a small epsilon to scales to prevent division by zero
    scales = scales + 1e-9

    # Compute (x - μᵢ) / sᵢ for all components
    standardized = (x - locations) / scales  # (batch_size, num_components)
    
    # Compute σ((x - μᵢ) / sᵢ) for all components
    logistic_cdfs = torch.sigmoid(standardized)  # (batch_size, num_components)
    
    # Logistic PDF = (1/s) * σ(z) * (1 - σ(z)) where z = (x - μ) / s
    logistic_pdfs = (1.0 / scales) * logistic_cdfs * (1.0 - logistic_cdfs)
    
    # Weighted sum: Σᵢ wᵢ * pdfᵢ(x)
    mixture_pdf = torch.sum(weights * logistic_pdfs, dim=1)  # (batch_size,)
    
    return mixture_pdf

def mixture_logistic_loss(mixture_weights, mixture_locations, mixture_scales, target_tp_ratio):
    """
    Negative log-likelihood loss for mixture of logistic distributions
    """
    # Clamp target to valid range [ε, 1-ε] for numerical stability
    target_clamped = torch.clamp(target_tp_ratio, 1e-6, 1-1e-6)
    
    # Compute PDF at target values
    pdf_values = mixture_logistic_pdf(target_clamped, mixture_weights, mixture_locations, mixture_scales)
    
    # Negative log-likelihood
    nll_loss = -torch.log(pdf_values + 1e-9).mean()

    return nll_loss

def precision_aware_loss(mixture_weights, mixture_locations, mixture_scales, 
                        target_precision, 
                        distribution_weight=1.0, precision_weight=0.1):
    """
    Combined loss with mixture of logistic distributions for TP ratio uncertainty
    """
    # Mixture logistic distribution loss (encourages proper uncertainty modeling)
    distribution_loss = mixture_logistic_loss(mixture_weights, mixture_locations, mixture_scales, target_precision)
    
    # Mean prediction loss (re-enabled with small weight to encourage sharper predictions)
    predicted_precision = torch.sum(mixture_weights * mixture_locations, dim=1)
    precision_loss = F.mse_loss(predicted_precision, target_precision)
    
    # Combined loss with weights
    total_loss = (distribution_weight * distribution_loss +
                  precision_weight * precision_loss)
    
    return total_loss, distribution_loss, precision_loss

def adversary_loss(adversary_pred, counts):
    """
    Loss for the adversary - tries to predict the number of alerts.
    """
    return F.mse_loss(adversary_pred.squeeze(), counts.float())

def calculate_predicted_precision(*args, **kwargs):
    """
    DEPRECATED: This function is now part of the loss function itself.
    This stub is kept for compatibility with older analysis scripts.
    """
    return 0.0 