#!/usr/bin/env python3
"""
Models package for precision-aware training
"""

from .dataset import CameraDensityDatasetPrecisionAware, collate_fn
from .architecture import ResidualMLP, MAB, DeepSetsPrecisionAware
from .loss_functions import (
    calculate_predicted_precision,
    mixture_logistic_cdf,
    mixture_logistic_pdf,
    mixture_logistic_loss,
    precision_aware_loss
)
from .training import (
    load_data,
    create_datasets,
    create_data_loaders,
    create_model_and_optimizer,
    setup_directories,
    train_epoch,
    validate_epoch,
    save_best_model,
    print_epoch_summary,
    print_training_setup,
    print_training_complete
)

__all__ = [
    # Dataset
    'CameraDensityDatasetPrecisionAware',
    'collate_fn',
    # Architecture
    'ResidualMLP',
    'MAB', 
    'DeepSetsPrecisionAware',
    # Loss functions
    'calculate_predicted_precision',
    'mixture_logistic_cdf',
    'mixture_logistic_pdf',
    'mixture_logistic_loss',
    'precision_aware_loss',
    # Training utilities
    'load_data',
    'create_datasets',
    'create_data_loaders',
    'create_model_and_optimizer',
    'setup_directories',
    'train_epoch',
    'validate_epoch',
    'save_best_model',
    'print_epoch_summary',
    'print_training_setup',
    'print_training_complete'
] 