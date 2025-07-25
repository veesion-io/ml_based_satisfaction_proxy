#!/usr/bin/env python3
"""
Distribution utilities for mixture of logistic distributions
"""

import numpy as np

def mixture_logistic_cdf_numpy(x, weights, locations, scales):
    """
    Numpy version of mixture logistic CDF: P(X ≤ x) = Σᵢ wᵢ * σ((x - μᵢ) / sᵢ)
    """
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip for numerical stability
    
    # Compute CDF for each component
    cdfs = sigmoid((x[:, None] - locations[None, :]) / scales[None, :])  # (n_points, n_components)
    
    # Weighted sum
    mixture_cdf = np.sum(weights[None, :] * cdfs, axis=1)  # (n_points,)
    
    return mixture_cdf

def mixture_logistic_pdf_numpy(x, weights, locations, scales):
    """
    Numpy version of mixture logistic PDF
    """
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    # Compute PDF for each component
    standardized = (x[:, None] - locations[None, :]) / scales[None, :]  # (n_points, n_components)
    sigmoid_vals = sigmoid(standardized)
    pdfs = (1.0 / scales[None, :]) * sigmoid_vals * (1.0 - sigmoid_vals)  # (n_points, n_components)
    
    # Weighted sum
    mixture_pdf = np.sum(weights[None, :] * pdfs, axis=1)  # (n_points,)
    
    return mixture_pdf

def find_quantile(cdf_x, cdf_y, q):
    """Find quantile q from CDF values"""
    idx = np.searchsorted(cdf_y, q)
    return cdf_x[min(idx, len(cdf_x)-1)]

def extract_tp_ratio_distribution_info(mixture_weights, mixture_locations, mixture_scales):
    """Extract statistics from mixture of logistic distributions"""
    
    # Mean of mixture = Σᵢ wᵢ * μᵢ (location parameters are means for logistic)
    mean = np.sum(mixture_weights * mixture_locations)
    
    # For variance: Var(mixture) = Σᵢ wᵢ * (μᵢ² + σᵢ²) - (Σᵢ wᵢ * μᵢ)²
    # For logistic distribution, variance = (π²/3) * s²
    logistic_variances = (np.pi**2 / 3) * (mixture_scales**2)
    second_moment = np.sum(mixture_weights * (mixture_locations**2 + logistic_variances))
    variance = second_moment - mean**2
    std = np.sqrt(max(variance, 0))  # Ensure non-negative
    
    # Compute CDF and find quantiles numerically
    x_eval = np.linspace(0, 1, 1000)  # Evaluation points
    cdf_vals = mixture_logistic_cdf_numpy(x_eval, mixture_weights, mixture_locations, mixture_scales)
    pdf_vals = mixture_logistic_pdf_numpy(x_eval, mixture_weights, mixture_locations, mixture_scales)
    
    # Mode (x with highest PDF)
    mode_idx = np.argmax(pdf_vals)
    mode = x_eval[mode_idx]
    
    # Quantiles
    ci_lower = find_quantile(x_eval, cdf_vals, 0.025)
    ci_upper = find_quantile(x_eval, cdf_vals, 0.975)
    p25 = find_quantile(x_eval, cdf_vals, 0.25)
    p75 = find_quantile(x_eval, cdf_vals, 0.75)
    median = find_quantile(x_eval, cdf_vals, 0.5)
    
    return {
        'mean': mean,
        'std': std,
        'variance': variance,
        'mode': mode,
        'median': median,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p25': p25,
        'p75': p75,
        'mixture_weights': mixture_weights,
        'mixture_locations': mixture_locations,
        'mixture_scales': mixture_scales,
        'pdf_x': x_eval,
        'pdf_y': pdf_vals,
        'cdf_x': x_eval,
        'cdf_y': cdf_vals
    } 