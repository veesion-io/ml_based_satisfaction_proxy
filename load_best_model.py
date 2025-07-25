#!/usr/bin/env python3
"""
Script to load the best model checkpoint and demonstrate usage
"""

import torch
import numpy as np
from density_prediction_training import DeepSetsAdvanced, N_BINS

def load_best_model(checkpoint_path="runs/best_model/best_checkpoint.pth"):
    """
    Load the best model from checkpoint
    
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
    model = DeepSetsAdvanced(
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
    
    print(f"✅ Best model loaded successfully!")
    print(f"   Epoch: {training_info['epoch']}")
    print(f"   R² Score: {training_info['r2_score']:.4f}")
    print(f"   TP R²: {training_info['r2_tp']:.4f}")
    print(f"   FP R²: {training_info['r2_fp']:.4f}")
    print(f"   Hyperparameters: {hyperparams}")
    
    return model, training_info

def predict_densities(model, camera_data, sample_size=30):
    """
    Predict TP/FP densities for a camera
    
    Args:
        model: Trained model
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
    
    # Add normalized sample size feature (using log scale like in training)
    max_training_size = 2000  # Should match SAMPLE_SIZE_RANGE[1] in training
    k_normalized = np.log(len(sample_data)) / np.log(max_training_size)
    k_feature = torch.full((len(sample_data), 1), fill_value=k_normalized, dtype=torch.float32)
    sample_features = torch.cat([base_features, k_feature], dim=1)
    
    # Add batch dimension and move to device
    sample_features = sample_features.unsqueeze(0).to(device)
    counts = torch.tensor([len(sample_data)], dtype=torch.float32).to(device)
    
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
    
    # Load the best model
    model, info = load_best_model()
    
    # Example: Load a camera file and predict densities
    camera_files = list(Path("data_by_camera").glob("*.parquet"))
    if camera_files:
        example_camera = pd.read_parquet(camera_files[0])
        print(f"\n📊 Example prediction for camera: {camera_files[0].name}")
        print(f"   Camera data shape: {example_camera.shape}")
        
        tp_density, fp_density = predict_densities(model, example_camera, sample_size=25)
        
        print(f"   Predicted TP density: {tp_density[:5]}... (sum: {tp_density.sum():.4f})")
        print(f"   Predicted FP density: {fp_density[:5]}... (sum: {fp_density.sum():.4f})")
        
        print("\n🎯 Model is ready for inference!")
    else:
        print("\n⚠️  No camera files found for example prediction") 