#!/usr/bin/env python3
"""
Sample-size-aware uncertainty calibration for convergence analysis
"""

import numpy as np

from .distribution_utils import extract_tp_ratio_distribution_info

class NoiseTuner:
    """
    Calibrates model's predicted uncertainty to properly reflect sample size dependency.
    The model doesn't properly learn that smaller samples should have wider uncertainty.
    This class applies theoretically-motivated uncertainty scaling.
    """
    
    def __init__(self, baseline_sample_size=1000, uncertainty_scaling_factor=1.0):
        """
        Initialize the uncertainty calibrator.
        Args:
            baseline_sample_size: Reference sample size for uncertainty calibration.
            uncertainty_scaling_factor: How much to scale uncertainty based on sample size.
        """
        self.baseline_sample_size = baseline_sample_size
        self.uncertainty_scaling_factor = uncertainty_scaling_factor

    def tune_noise(self, uncertainty_info, sample_size, max_sample_size=2000):
        """
        Apply sample-size-aware uncertainty calibration.
        
        The key insight: For binomial processes, uncertainty scales as 1/sqrt(n).
        We apply this scaling to the mixture scales to get proper uncertainty propagation.
        """
        
        # Calculate theoretical uncertainty scaling factor
        # Uncertainty should scale as 1/sqrt(sample_size)
        baseline_uncertainty = 1.0 / np.sqrt(self.baseline_sample_size)
        current_uncertainty = 1.0 / np.sqrt(max(sample_size, 1))
        
        # Scale factor: how much more uncertain should this be relative to baseline
        uncertainty_ratio = current_uncertainty / baseline_uncertainty
        
        # Apply scaling to mixture scales - much more aggressive now
        scale_multiplier = 1.0 + (uncertainty_ratio - 1.0) * self.uncertainty_scaling_factor
        
        # Ensure reasonable bounds but allow much wider range
        scale_multiplier = np.clip(scale_multiplier, 0.2, 10.0)
        
        # Apply the scaling to mixture scales
        adjusted_scales = uncertainty_info['mixture_scales'] * scale_multiplier
        
        # Re-calculate distribution statistics with the adjusted scales
        adjusted_info = extract_tp_ratio_distribution_info(
            uncertainty_info['mixture_weights'],
            uncertainty_info['mixture_locations'],
            adjusted_scales
        )
        
        return adjusted_info 