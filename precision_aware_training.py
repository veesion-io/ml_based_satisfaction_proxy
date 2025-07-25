#!/usr/bin/env python3
"""
Precision-Aware Training with Biased Ratio Loss
Implements a loss that optimizes for precision accuracy rather than ratio accuracy
"""

import torch
import argparse
import warnings
from models import (
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

warnings.filterwarnings('ignore')

# Default hyperparameters
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 22
DEFAULT_LEARNING_RATE = 0.0005793892334263356
DEFAULT_PHI_DIM = 208
DEFAULT_NUM_HEADS = 7
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_WEIGHT_DECAY = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_training_loop(model, train_loader, val_loader, optimizer, args, best_model_path):
    """Run the main training loop"""
    best_precision_score = -1.0
    
    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        train_losses = train_epoch(model, train_loader, optimizer, args, DEVICE)
        
        # Validate for one epoch
        val_losses = validate_epoch(model, val_loader, args, DEVICE)
        
        # Check if this is the best model so far
        precision_score = -val_losses[3]  # Negative validation precision loss
        is_best = precision_score > best_precision_score
        
        if is_best:
            best_precision_score = precision_score
            save_best_model(model, optimizer, epoch, precision_score, 
                           train_losses[0], val_losses[0], 
                           train_losses[3], val_losses[3], args, best_model_path)
        
        # Print epoch summary
        print_epoch_summary(epoch, train_losses, val_losses, is_best)
    
    return best_precision_score

def main(args):
    """Main training function - now much shorter and more readable"""
    # Print setup information
    print_training_setup(DEVICE, args)
    
    # Load and prepare data
    train_data, val_data = load_data()
    train_ds, val_ds = create_datasets(train_data, val_data)
    train_loader, val_loader = create_data_loaders(train_ds, val_ds, args.batch_size)
    
    # Create model and optimizer
    model, optimizer = create_model_and_optimizer(args, DEVICE)
    
    # Setup directories
    plots_dir, best_model_path = setup_directories()
    
    # Run training loop
    best_precision_score = run_training_loop(model, train_loader, val_loader, 
                                           optimizer, args, best_model_path)
    
    # Print completion message
    print_training_complete(best_model_path, best_precision_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--phi_dim', type=int, default=DEFAULT_PHI_DIM)
    parser.add_argument('--num_heads', type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument('--dropout_rate', type=float, default=DEFAULT_DROPOUT_RATE, help='Dropout rate for regularization')
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY, help='Weight decay for optimizer regularization')
    parser.add_argument('--density_weight', type=float, default=1.0, help='Weight for density loss')
    parser.add_argument('--distribution_weight', type=float, default=2.0, help='Weight for Beta distribution loss')
    parser.add_argument('--precision_weight', type=float, default=1.0, help='Weight for mean precision loss')
    main(parser.parse_args()) 