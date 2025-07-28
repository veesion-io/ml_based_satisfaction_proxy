#!/usr/bin/env python3
"""
Training utilities for precision-aware training
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle

from .dataset import CameraDensityDatasetPrecisionAware, collate_fn
from .architecture import DeepSetsPrecisionAware
from .loss_functions import precision_aware_loss

# Constants
N_BINS = 20
SAMPLE_SIZE_RANGE = (10, 2000)

def load_data():
    """Load and split the data"""
    with open("ground_truth_histograms.pkl", 'rb') as f: 
        data = pickle.load(f)
    
    # Ground truth precision is now pre-calculated
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, val_data

def create_datasets(train_data, val_data):
    """Create training and validation datasets"""
    train_ds = CameraDensityDatasetPrecisionAware(train_data, SAMPLE_SIZE_RANGE)
    val_ds = CameraDensityDatasetPrecisionAware(val_data, SAMPLE_SIZE_RANGE)
    return train_ds, val_ds

def create_data_loaders(train_ds, val_ds, batch_size):
    """Create data loaders for training and validation"""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                           num_workers=4, collate_fn=collate_fn)
    return train_loader, val_loader

def create_model_and_optimizer(args, device):
    """Create model and optimizer"""
    model = DeepSetsPrecisionAware(args.phi_dim, N_BINS, args.num_heads, args.dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    return model, optimizer

def setup_directories():
    """Set up necessary directories"""
    plots_dir = "plots_precision_aware"
    os.makedirs(plots_dir, exist_ok=True)
    
    best_model_path = "runs/best_model/best_checkpoint_precision_aware.pth"
    os.makedirs("runs/best_model", exist_ok=True)
    
    return plots_dir, best_model_path

def train_epoch(model, train_loader, optimizer, args, device):
    """Train model for one epoch"""
    model.train()
    total_train_loss = 0
    total_train_distribution_loss = 0
    total_train_precision_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        if batch[0] is None: 
            continue
        
        features, gt_precision, counts = [b.to(device) for b in batch]
        optimizer.zero_grad()
        mixture_weights, mixture_locations, mixture_scales = model(features, counts)
        
        total_loss, distribution_loss, precision_loss = precision_aware_loss(
            mixture_weights, mixture_locations, mixture_scales, 
            gt_precision,
            distribution_weight=args.distribution_weight, 
            precision_weight=args.precision_weight
        )
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_train_loss += total_loss.item()
        total_train_distribution_loss += distribution_loss.item()
        total_train_precision_loss += precision_loss.item()
    
    n_batches = len(train_loader)
    return total_train_loss / n_batches, total_train_distribution_loss / n_batches, total_train_precision_loss / n_batches

def validate_epoch(model, val_loader, args, device):
    """Validate model for one epoch"""
    model.eval()
    total_val_loss = 0
    total_val_distribution_loss = 0
    total_val_precision_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            if batch[0] is None: 
                continue
            
            features, gt_precision, counts = [b.to(device) for b in batch]
            mixture_weights, mixture_locations, mixture_scales = model(features, counts)
            
            total_loss, distribution_loss, precision_loss = precision_aware_loss(
                mixture_weights, mixture_locations, mixture_scales, 
                gt_precision,
                distribution_weight=args.distribution_weight, 
                precision_weight=args.precision_weight
            )
            
            total_val_loss += total_loss.item()
            total_val_distribution_loss += distribution_loss.item()
            total_val_precision_loss += precision_loss.item()
    
    n_batches = len(val_loader)
    return total_val_loss / n_batches, total_val_distribution_loss / n_batches, total_val_precision_loss / n_batches

def save_best_model(model, optimizer, epoch, distribution_loss, train_loss, val_loss, 
                   train_distribution_loss, val_distribution_loss, args, best_model_path):
    """Save the best model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'distribution_loss': distribution_loss,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_distribution_loss': train_distribution_loss,
        'val_distribution_loss': val_distribution_loss,
        'hyperparameters': {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'phi_dim': args.phi_dim,
            'num_heads': args.num_heads,
            'epochs': args.epochs,
            'distribution_weight': args.distribution_weight,
            'precision_weight': args.precision_weight,
            'dropout_rate': args.dropout_rate,
            'weight_decay': args.weight_decay
        }
    }, best_model_path)

def print_epoch_summary(epoch, train_losses, val_losses, is_best=False):
    """Print training summary for an epoch"""
    train_loss, train_distribution_loss, train_precision_loss = train_losses
    val_loss, val_distribution_loss, val_precision_loss = val_losses
    
    status = "✅ BEST MODEL SAVED" if is_best else ""
    print(f"Epoch {epoch} | Train: {train_loss:.4f} (D:{train_distribution_loss:.4f}, P:{train_precision_loss:.4f}) | "
          f"Val: {val_loss:.4f} (D:{val_distribution_loss:.4f}, P:{val_precision_loss:.4f}) | {status}")

def print_training_setup(device, args):
    """Print training setup information"""
    print(f"Starting precision-aware training on {device}...")
    print(f"Hyperparameters: LR={args.learning_rate}, Dropout={args.dropout_rate}, Weight Decay={args.weight_decay}")

def print_training_complete(best_model_path, best_precision_score):
    """Print training completion message"""
    print(f"\n🎉 Precision-aware training complete!")
    print(f"Best model saved to: {best_model_path} (Precision Score = {best_precision_score:.4f})") 