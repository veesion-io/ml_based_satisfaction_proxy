#!/usr/bin/env python3
"""
Hyperparameter Tuning for Precision-Aware Training
Uses Optuna for Bayesian optimization to find optimal hyperparameters
"""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import pickle
import argparse
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

# Import from the new modular structure
from models import (
    CameraDensityDatasetPrecisionAware,
    DeepSetsPrecisionAware,
    precision_aware_loss,
    collate_fn
)

# Constants
SAMPLE_SIZE_RANGE = (10, 2000)
N_BINS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings('ignore')

def objective(trial, train_data, val_data, n_epochs=10):
    """Objective function for Optuna optimization"""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [2, 4, 8, 16])
    phi_dim = trial.suggest_categorical('phi_dim', [128, 208, 256, 320, 384])
    num_heads = trial.suggest_categorical('num_heads', [4, 7, 8, 16])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    
    # Loss weights
    density_weight = trial.suggest_float('density_weight', 0.1, 3.0)
    distribution_weight = trial.suggest_float('distribution_weight', 0.1, 5.0)
    precision_weight = trial.suggest_float('precision_weight', 0.1, 3.0)
    
    # Ensure phi_dim is divisible by num_heads
    if phi_dim % num_heads != 0:
        phi_dim = (phi_dim // num_heads + 1) * num_heads
    
    try:
        # Create datasets and dataloaders
        train_ds = CameraDensityDatasetPrecisionAware(train_data, SAMPLE_SIZE_RANGE)
        val_ds = CameraDensityDatasetPrecisionAware(val_data, SAMPLE_SIZE_RANGE)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                 num_workers=2, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                               num_workers=2, collate_fn=collate_fn)
        
        # Create model and optimizer
        model = DeepSetsPrecisionAware(phi_dim, N_BINS, num_heads, dropout_rate).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Training loop
        best_val_precision_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Training
            model.train()
            total_train_loss = 0
            train_batches = 0
            
            for batch in train_loader:
                if batch[0] is None: 
                    continue
                
                features, gt_tp, gt_fp, gt_ratio, gt_precision, counts = [b.to(DEVICE) for b in batch]
                optimizer.zero_grad()
                pred_tp, pred_fp, mixture_weights, mixture_locations, mixture_scales = model(features, counts)
                
                total_loss, density_loss, distribution_loss, precision_loss = precision_aware_loss(
                    pred_tp, pred_fp, mixture_weights, mixture_locations, mixture_scales, 
                    gt_tp, gt_fp, gt_ratio, gt_precision, counts,
                    density_weight=density_weight, 
                    distribution_weight=distribution_weight, 
                    precision_weight=precision_weight
                )
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_train_loss += total_loss.item()
                train_batches += 1
            
            # Validation
            model.eval()
            total_val_precision_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if batch[0] is None: 
                        continue
                    
                    features, gt_tp, gt_fp, gt_ratio, gt_precision, counts = [b.to(DEVICE) for b in batch]
                    pred_tp, pred_fp, mixture_weights, mixture_locations, mixture_scales = model(features, counts)
                    
                    total_loss, density_loss, distribution_loss, precision_loss = precision_aware_loss(
                        pred_tp, pred_fp, mixture_weights, mixture_locations, mixture_scales, 
                        gt_tp, gt_fp, gt_ratio, gt_precision, counts,
                        density_weight=density_weight, 
                        distribution_weight=distribution_weight, 
                        precision_weight=precision_weight
                    )
                    
                    total_val_precision_loss += precision_loss.item()
                    val_batches += 1
            
            avg_val_precision_loss = total_val_precision_loss / max(val_batches, 1)
            
            # Early stopping based on precision loss
            if avg_val_precision_loss < best_val_precision_loss:
                best_val_precision_loss = avg_val_precision_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            # Report intermediate value for pruning
            trial.report(avg_val_precision_loss, epoch)
            
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return best_val_precision_loss
    
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('inf')

def run_hyperparameter_tuning(args):
    """Main hyperparameter tuning function"""
    
    # Load data
    print("Loading data...")
    with open("ground_truth_histograms.pkl", 'rb') as f:
        data = pickle.load(f)
    
    # Split data
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Training on {len(train_data)} cameras, validating on {len(val_data)} cameras")
    
    # Create study
    study_name = f"precision_aware_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage_url = f"sqlite:///hyperparameter_studies/{study_name}.db"
    os.makedirs("hyperparameter_studies", exist_ok=True)
    
    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )
    
    print(f"Starting hyperparameter optimization with {args.n_trials} trials...")
    print(f"Using {args.tuning_epochs} epochs per trial")
    print(f"Study will be saved to: {storage_url}")
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, train_data, val_data, args.tuning_epochs),
        n_trials=args.n_trials,
        timeout=args.timeout if args.timeout > 0 else None,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation precision loss: {study.best_value:.6f}")
    
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best parameters
    results_dir = "tuning_results"
    os.makedirs(results_dir, exist_ok=True)
    
    best_params_file = f"{results_dir}/best_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'study_name': study_name,
        'n_trials': len(study.trials),
        'tuning_epochs': args.tuning_epochs
    }
    
    with open(best_params_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBest parameters saved to: {best_params_file}")
    
    # Generate training command
    print("\n" + "="*80)
    print("RECOMMENDED TRAINING COMMAND")
    print("="*80)
    
    cmd_parts = ["python precision_aware_training.py"]
    for key, value in study.best_params.items():
        cmd_parts.append(f"--{key} {value}")
    cmd_parts.append("--epochs 50")  # Use more epochs for final training
    
    print(" \\\n  ".join(cmd_parts))
    
    # Save command to file
    cmd_file = f"{results_dir}/best_training_command.sh"
    with open(cmd_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Best hyperparameters found by tuning\n")
        f.write(" \\\n  ".join(cmd_parts))
        f.write("\n")
    
    print(f"\nTraining command saved to: {cmd_file}")
    
    return study.best_params, study.best_value

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for precision-aware training")
    parser.add_argument('--n_trials', type=int, default=50, 
                       help='Number of trials for optimization')
    parser.add_argument('--tuning_epochs', type=int, default=10,
                       help='Number of epochs per trial (use fewer for faster tuning)')
    parser.add_argument('--timeout', type=int, default=0,
                       help='Timeout in seconds (0 = no timeout)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PRECISION-AWARE MODEL HYPERPARAMETER TUNING")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Trials: {args.n_trials}")
    print(f"Epochs per trial: {args.tuning_epochs}")
    if args.timeout > 0:
        print(f"Timeout: {args.timeout} seconds")
    print("="*80)
    
    best_params, best_value = run_hyperparameter_tuning(args)
    
    print(f"\n🎉 Hyperparameter tuning complete!")
    print(f"Best validation precision loss: {best_value:.6f}")

if __name__ == "__main__":
    main() 