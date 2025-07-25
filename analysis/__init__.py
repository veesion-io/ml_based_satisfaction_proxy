#!/usr/bin/env python3
"""
Analysis package for average precision convergence analysis
"""

from .model_loader import (
    load_precision_aware_model,
    predict_densities_and_ratio_precision_aware,
    calculate_simple_tp_ratio,
    predict_average_precision_aware,
    predict_average_precision_aware_with_uncertainty
)

from .distribution_utils import (
    mixture_logistic_cdf_numpy,
    mixture_logistic_pdf_numpy,
    extract_tp_ratio_distribution_info
)

from .processing import (
    load_camera_paths,
    setup_multiprocessing,
    prepare_multiprocessing_args,
    process_camera_chunk,
    run_parallel_processing
)

from .aggregation import (
    aggregate_results_by_proportion,
    save_results
)

from .plotting import (
    create_beautiful_quantile_plot
)

from .summary import (
    print_average_precision_summary
)

__all__ = [
    # Model loading
    'load_precision_aware_model',
    'predict_densities_and_ratio_precision_aware',
    'calculate_simple_tp_ratio',
    'predict_average_precision_aware',
    'predict_average_precision_aware_with_uncertainty',
    
    # Distribution utilities
    'mixture_logistic_cdf_numpy',
    'mixture_logistic_pdf_numpy',
    'extract_tp_ratio_distribution_info',
    
    # Processing
    'load_camera_paths',
    'setup_multiprocessing',
    'prepare_multiprocessing_args',
    'process_camera_chunk',
    'run_parallel_processing',
    
    # Aggregation
    'aggregate_results_by_proportion',
    'save_results',
    
    # Plotting
    'create_beautiful_quantile_plot',
    
    # Summary
    'print_average_precision_summary'
] 