#!/usr/bin/env python3
"""
Script to load the improved model checkpoint and demonstrate usage
Handles full range of sample sizes (10-2000 alerts)
"""

import torch
import numpy as np
from density_prediction_training_improved import DeepSetsAdvancedImproved, N_BINS

def load_best_model_improved(checkpoint_path="runs/best_model/best_checkpoint_improved.pth"):
    """
    Load the improved model from checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
    
    Returns:
        model: Loaded model ready for inference
        checkpoint_info: Dictionary with training information
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract hyperparameters
    hyperparams = checkpoint['hyperparameters']
    
    # Initialize model with correct architecture
    model = DeepSetsAdvancedImproved(
        phi_dim=hyperparams['phi_dim'],
        n_bins=N_BINS,
        num_heads=hyperparams['num_heads']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Extract training info
    training_info = {
        'epoch': checkpoint['epoch'],
        'r2_score': checkpoint['r2_score'],
        'r2_tp': checkpoint['r2_tp'],
        'r2_fp': checkpoint['r2_fp'],
        'train_loss': checkpoint['train_loss'],
        'val_loss': checkpoint['val_loss'],
        'hyperparameters': hyperparams
    }
    
    print(f"✅ Improved model loaded successfully!")
    print(f"   Epoch: {training_info['epoch']}")
    print(f"   R² Score: {training_info['r2_score']:.4f}")
    print(f"   TP R²: {training_info['r2_tp']:.4f}")
    print(f"   FP R²: {training_info['r2_fp']:.4f}")
    print(f"   Sample Range: {hyperparams.get('sample_range', 'Unknown')}")
    print(f"   Hyperparameters: {hyperparams}")
    
    return model, training_info

def predict_densities_improved(model, camera_data, sample_size=30):
    """
    Predict TP/FP densities using the improved model
    
    Args:
        model: Trained improved model
        camera_data: DataFrame with 'max_proba' and 'is_theft' columns
        sample_size: Number of alerts to sample
    
    Returns:
        tp_density: Predicted TP probability density
        fp_density: Predicted FP probability density
    """
    device = next(model.parameters()).device
    
    # Sample data
    if len(camera_data) < sample_size:
        sample_data = camera_data
    else:
        sample_data = camera_data.sample(n=sample_size, replace=False)
    
    # Prepare input features
    base_features = torch.tensor(
        sample_data[['max_proba', 'is_theft']].values, 
        dtype=torch.float32
    )
    
    # Multi-scale size features (matching training)
    max_training_size = 2000  # Should match SAMPLE_SIZE_RANGE[1] in training
    camera_size = len(camera_data)
    k = len(sample_data)
    
    # Calculate size percentiles (approximation for inference)
    size_percentiles = np.array([330, 656, 1110, 1986, 2356])  # From earlier analysis
    
    # Linear normalization (0-1)
    k_linear = k / max_training_size
    # Log normalization (0-1)  
    k_log = np.log(k) / np.log(max_training_size)
    # Percentile-based normalization
    k_percentile = np.searchsorted(size_percentiles, k) / len(size_percentiles)
    
    # Combine multiple size features
    size_features = torch.tensor([k_linear, k_log, k_percentile], dtype=torch.float32)
    k_feature = size_features.unsqueeze(0).repeat(k, 1)
    
    sample_features = torch.cat([base_features, k_feature], dim=1)
    
    # Add batch dimension and move to device
    sample_features = sample_features.unsqueeze(0).to(device)
    counts = torch.tensor([k], dtype=torch.float32).to(device)
    
    # Predict
    with torch.no_grad():
        tp_pred, fp_pred = model(sample_features, counts)
    
    # Convert to numpy
    tp_density = tp_pred.squeeze().cpu().numpy()
    fp_density = fp_pred.squeeze().cpu().numpy()
    
    return tp_density, fp_density

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from pathlib import Path
    
    # Load the improved model
    try:
        model, info = load_best_model_improved()
        
        # Example: Load a camera file and predict densities at different sample sizes
        camera_files = list(Path("data_by_camera").glob("*.parquet"))
        if camera_files:
            example_camera = pd.read_parquet(camera_files[0])
            print(f"\n📊 Example prediction for camera: {camera_files[0].name}")
            print(f"   Camera data shape: {example_camera.shape}")
            
            # Test different sample sizes
            test_sizes = [25, 50, 100, 500, len(example_camera)]
            
            print(f"\n🧪 Testing different sample sizes:")
            for size in test_sizes:
                if size <= len(example_camera):
                    tp_density, fp_density = predict_densities_improved(model, example_camera, sample_size=size)
                    
                    print(f"   {size:4d} alerts → TP sum: {tp_density.sum():.4f}, FP sum: {fp_density.sum():.4f}")
                    
            print("\n🎯 Improved model is ready for inference across all sample sizes!")
        else:
            print("\n⚠️  No camera files found for example prediction")
            
    except FileNotFoundError:
        print("⚠️  Improved model checkpoint not found. Train the improved model first using:")
        print("   python density_prediction_training_improved.py") 