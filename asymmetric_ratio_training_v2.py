#!/usr/bin/env python3
"""
Refined Asymmetric Ratio Training - Version 2
Penalize UNDER-prediction of TP ratios with subtle penalties
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
import optuna
from sklearn.model_selection import train_test_split
from density_prediction_training_with_ratio import ResidualMLP, MAB

class DeepSetsRefinedAsymmetric(nn.Module):
    def __init__(self, phi_dim, n_bins, num_heads):
        super(DeepSetsRefinedAsymmetric, self).__init__()
        
        # Ensure phi_dim is divisible by num_heads
        if phi_dim % num_heads != 0:
            phi_dim = (phi_dim // num_heads + 1) * num_heads
        
        # Feature extraction
        self.phi = nn.Sequential(
            nn.Linear(3, phi_dim),  # [prob, is_theft, normalized_k]
            ResidualMLP(phi_dim),
            ResidualMLP(phi_dim)
        )
        
        # Multi-head attention pooling
        self.pooling = MAB(phi_dim, num_heads)
        self.query = nn.Parameter(torch.randn(1, 1, phi_dim))
        
        # Final processing
        self.rho = nn.Sequential(
            ResidualMLP(phi_dim),
            ResidualMLP(phi_dim)
        )
        
        # Output heads
        self.tp_head = nn.Linear(phi_dim, n_bins)
        self.fp_head = nn.Linear(phi_dim, n_bins)
        self.ratio_head = nn.Linear(phi_dim, 1)
    
    def forward(self, x, counts):
        # Create mask for valid sequence elements
        mask = torch.arange(x.size(1), device=x.device)[None, :] < counts[:, None]
        
        # Feature extraction
        phi_out = self.phi(x * mask.unsqueeze(-1))
        
        # Attention pooling
        query = self.query.repeat(x.shape[0], 1, 1)
        agg = self.pooling(query, phi_out).squeeze(1)
        
        # Final processing
        rho_out = self.rho(agg)
        
        # Output predictions
        tp_out = F.softmax(self.tp_head(rho_out), dim=1)
        fp_out = F.softmax(self.fp_head(rho_out), dim=1)
        ratio_out = torch.sigmoid(self.ratio_head(rho_out)).squeeze(-1)
        
        return tp_out, fp_out, ratio_out

def refined_asymmetric_loss(predictions, targets, under_penalty=1.5, over_penalty=1.0):
    """
    Refined asymmetric loss that gently penalizes under-prediction of TP ratios
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets  
        under_penalty: Penalty multiplier for under-prediction (should be > 1.0)
        over_penalty: Penalty multiplier for over-prediction (should be 1.0)
    """
    tp_pred, fp_pred, ratio_pred = predictions
    tp_target, fp_target, ratio_target = targets
    
    # Standard losses for densities
    tp_loss = F.kl_div(torch.log(tp_pred + 1e-9), tp_target, reduction='batchmean')
    fp_loss = F.kl_div(torch.log(fp_pred + 1e-9), fp_target, reduction='batchmean')
    
    # Refined asymmetric loss for ratio
    ratio_error = ratio_pred - ratio_target
    
    # Apply different penalties based on error direction
    under_mask = ratio_error < 0  # Under-prediction (bad for precision estimation)
    over_mask = ratio_error >= 0  # Over-prediction (less problematic)
    
    ratio_loss = torch.zeros_like(ratio_error)
    ratio_loss[under_mask] = under_penalty * (ratio_error[under_mask] ** 2)
    ratio_loss[over_mask] = over_penalty * (ratio_error[over_mask] ** 2)
    ratio_loss = ratio_loss.mean()
    
    return tp_loss, fp_loss, ratio_loss

def prepare_training_sample(camera_data, sample_size_range=(50, 2000)):
    """Prepare a single training sample from camera data"""
    # Load camera data
    alerts_df = pd.read_parquet(camera_data['file_path'])
    
    if len(alerts_df) < sample_size_range[0]:
        return None
    
    # Sample random subset
    k = min(len(alerts_df), np.random.randint(*sample_size_range))
    sampled = alerts_df.sample(n=k, replace=False)
    
    # Prepare features: [prob, is_theft, normalized_k]
    probs = sampled['max_proba'].values
    is_theft = sampled['is_theft'].values
    k_normalized = np.log(k) / np.log(2000)
    
    # Create feature matrix and pad to max length
    max_len = 2000
    features = np.zeros((max_len, 3))
    features[:k, 0] = probs
    features[:k, 1] = is_theft
    features[:k, 2] = k_normalized
    
    # Get target densities
    tp_density = camera_data['tp_density']
    fp_density = camera_data['fp_density']
    
    # Compute CORRECT TP ratio from actual alert counts in the FULL camera data
    tp_count = len(alerts_df[alerts_df['is_theft'] == 1])
    total_count = len(alerts_df)
    tp_ratio = tp_count / total_count if total_count > 0 else 0.0
    
    return {
        'features': torch.FloatTensor(features),
        'tp_histogram': torch.FloatTensor(tp_density),  # Use density as histogram
        'fp_histogram': torch.FloatTensor(fp_density),
        'tp_ratio': tp_ratio,
        'count': k
    }

def train_refined_asymmetric_model(trial=None):
    """Train the refined asymmetric model"""
    
    # Hyperparameters
    if trial:
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [2, 4, 8])
        phi_dim = trial.suggest_categorical('phi_dim', [128, 208, 256])
        num_heads = trial.suggest_categorical('num_heads', [4, 7, 8])
        epochs = trial.suggest_int('epochs', 10, 25)
        under_penalty = trial.suggest_float('under_penalty', 1.2, 2.0)  # Subtle penalties
        over_penalty = 1.0  # Keep over-prediction penalty at 1.0
    else:
        # Use best hyperparameters from previous training
        learning_rate = 0.0005793892334263356
        batch_size = 4
        phi_dim = 208
        num_heads = 7
        epochs = 20
        under_penalty = 1.3  # Subtle under-prediction penalty
        over_penalty = 1.0
    
    # Load training data
    print("🔄 Loading and preparing training data...")
    with open("ground_truth_histograms.pkl", 'rb') as f:
        data = pickle.load(f)
    
    # Split into train/val
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    print(f"🔧 Training refined asymmetric model...")
    print(f"   Under-penalty: {under_penalty}x, Over-penalty: {over_penalty}x")
    print(f"   Batch size: {batch_size}, Learning rate: {learning_rate:.6f}")
    print(f"   Training cameras: {len(train_data)}, Validation: {len(val_data)}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepSetsRefinedAsymmetric(phi_dim, 20, num_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_losses = {'tp': [], 'fp': [], 'ratio': []}
        
        # Generate training batches on the fly
        num_batches = len(train_data) // batch_size
        
        for batch_idx in range(num_batches):
            # Sample cameras for this batch
            batch_cameras = np.random.choice(train_data, size=batch_size, replace=False)
            
            # Prepare batch
            batch_samples = []
            for camera in batch_cameras:
                sample = prepare_training_sample(camera)
                if sample is not None:
                    batch_samples.append(sample)
            
            if len(batch_samples) == 0:
                continue
                
            # Convert to tensors
            features_batch = torch.stack([s['features'] for s in batch_samples]).to(device)
            tp_targets = torch.stack([s['tp_histogram'] for s in batch_samples]).to(device)
            fp_targets = torch.stack([s['fp_histogram'] for s in batch_samples]).to(device)
            ratio_targets = torch.tensor([s['tp_ratio'] for s in batch_samples], dtype=torch.float32).to(device)
            counts = torch.tensor([s['count'] for s in batch_samples], dtype=torch.long).to(device)
            
            # Forward pass
            tp_pred, fp_pred, ratio_pred = model(features_batch, counts)
            
            # Calculate refined asymmetric loss
            predictions = (tp_pred, fp_pred, ratio_pred)
            targets = (tp_targets, fp_targets, ratio_targets)
            tp_loss, fp_loss, ratio_loss = refined_asymmetric_loss(
                predictions, targets, under_penalty, over_penalty
            )
            
            # Total loss with balanced weighting
            total_loss = tp_loss + fp_loss + ratio_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track losses
            epoch_losses['tp'].append(tp_loss.item())
            epoch_losses['fp'].append(fp_loss.item())
            epoch_losses['ratio'].append(ratio_loss.item())
        
        # Print progress
        if epoch % 5 == 0 or epoch == epochs - 1:
            avg_tp_loss = np.mean(epoch_losses['tp'])
            avg_fp_loss = np.mean(epoch_losses['fp'])
            avg_ratio_loss = np.mean(epoch_losses['ratio'])
            total_avg = avg_tp_loss + avg_fp_loss + avg_ratio_loss
            
            print(f"Epoch {epoch+1:2d}: Total={total_avg:.4f} "
                  f"(TP={avg_tp_loss:.4f}, FP={avg_fp_loss:.4f}, Ratio={avg_ratio_loss:.4f})")
    
    # Evaluate model
    print("🧪 Evaluating model...")
    r2_score, r2_tp, r2_fp, r2_ratio = evaluate_model(model, val_data[:50])  # Quick eval
    
    print(f"✅ Training complete!")
    print(f"   Overall R²: {r2_score:.4f}")
    print(f"   TP R²: {r2_tp:.4f}, FP R²: {r2_fp:.4f}, Ratio R²: {r2_ratio:.4f}")
    
    # Save model
    if not trial:  # Only save when not in hyperparameter search
        save_path = "runs/best_model/best_checkpoint_refined_asymmetric.pth"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'r2_score': r2_score,
            'r2_tp': r2_tp,
            'r2_fp': r2_fp,
            'r2_ratio': r2_ratio,
            'hyperparameters': {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'phi_dim': phi_dim,
                'num_heads': num_heads,
                'epochs': epochs,
                'under_penalty': under_penalty,
                'over_penalty': over_penalty
            }
        }, save_path)
        print(f"💾 Model saved to {save_path}")
    
    return r2_score

def evaluate_model(model, test_data):
    """Evaluate model performance"""
    model.eval()
    device = next(model.parameters()).device
    
    tp_predictions = []
    fp_predictions = []
    ratio_predictions = []
    tp_targets = []
    fp_targets = []
    ratio_targets = []
    
    with torch.no_grad():
        for camera_data in test_data:
            sample = prepare_training_sample(camera_data, sample_size_range=(100, 1000))
            if sample is None:
                continue
                
            features = sample['features'].unsqueeze(0).to(device)
            count = torch.tensor([sample['count']], dtype=torch.long).to(device)
            
            tp_pred, fp_pred, ratio_pred = model(features, count)
            
            tp_predictions.append(tp_pred.cpu().numpy()[0])
            fp_predictions.append(fp_pred.cpu().numpy()[0])
            ratio_predictions.append(ratio_pred.cpu().item())
            
            # Use the original camera data for targets
            tp_targets.append(camera_data['tp_density'])
            fp_targets.append(camera_data['fp_density'])
            
            # Compute CORRECT ground truth ratio from actual alert counts
            alerts_df = pd.read_parquet(camera_data['file_path'])
            tp_count = len(alerts_df[alerts_df['is_theft'] == 1])
            total_count = len(alerts_df)
            gt_ratio = tp_count / total_count if total_count > 0 else 0.0
            ratio_targets.append(gt_ratio)
    
    if len(tp_predictions) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Calculate R² scores
    from sklearn.metrics import r2_score
    
    tp_predictions = np.array(tp_predictions)
    fp_predictions = np.array(fp_predictions)
    ratio_predictions = np.array(ratio_predictions)
    tp_targets = np.array(tp_targets)
    fp_targets = np.array(fp_targets)
    ratio_targets = np.array(ratio_targets)
    
    # Flatten for R² calculation
    tp_flat_pred = tp_predictions.flatten()
    tp_flat_target = tp_targets.flatten()
    fp_flat_pred = fp_predictions.flatten()
    fp_flat_target = fp_targets.flatten()
    
    r2_tp = r2_score(tp_flat_target, tp_flat_pred)
    r2_fp = r2_score(fp_flat_target, fp_flat_pred)
    r2_ratio = r2_score(ratio_targets, ratio_predictions)
    
    # Overall R² (weighted by importance)
    r2_overall = (r2_tp + r2_fp + r2_ratio) / 3
    
    return r2_overall, r2_tp, r2_fp, r2_ratio

if __name__ == "__main__":
    print("🚀 REFINED ASYMMETRIC RATIO TRAINING V2")
    print("=" * 50)
    print("Strategy: Gently penalize UNDER-prediction of TP ratios")
    print("Hypothesis: Under-predicted ratios → under-estimated precision")
    print()
    
    # Train the refined model
    final_score = train_refined_asymmetric_model()
    
    print(f"\n🎯 Final R² Score: {final_score:.4f}")
    print("🔍 Next: Test this model against the baseline!") 