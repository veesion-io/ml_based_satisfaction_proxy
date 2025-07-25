#!/usr/bin/env python3
"""
Precision-Calibrated Training
Keep the density+ratio approach but add a calibration layer for precision correction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from density_prediction_training_with_ratio import ResidualMLP, MAB

class DeepSetsPrecisionCalibrated(nn.Module):
    def __init__(self, phi_dim, n_bins, num_heads):
        super(DeepSetsPrecisionCalibrated, self).__init__()
        
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
        
        # Output heads for densities and ratio
        self.tp_head = nn.Linear(phi_dim, n_bins)
        self.fp_head = nn.Linear(phi_dim, n_bins)
        self.ratio_head = nn.Linear(phi_dim, 1)
        
        # NEW: Precision calibration layer
        # Takes raw precision estimates and corrects systematic bias
        self.precision_calibrator = nn.Sequential(
            nn.Linear(n_bins + 1, phi_dim // 2),  # precision_per_bin + sample_size_feature
            nn.ReLU(),
            nn.Linear(phi_dim // 2, phi_dim // 4),
            nn.ReLU(),
            nn.Linear(phi_dim // 4, n_bins)  # Calibrated precision per bin
        )
    
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
        
        # Standard output predictions
        tp_out = F.softmax(self.tp_head(rho_out), dim=1)
        fp_out = F.softmax(self.fp_head(rho_out), dim=1)
        ratio_out = torch.sigmoid(self.ratio_head(rho_out)).squeeze(-1)
        
        # Calculate raw precision per bin
        fp_ratio = 1.0 - ratio_out.unsqueeze(-1)
        tp_scaled = tp_out * ratio_out.unsqueeze(-1)
        fp_scaled = fp_out * fp_ratio
        raw_precision = tp_scaled / (tp_scaled + fp_scaled + 1e-9)
        
        # Create sample size feature for calibration
        sample_size_feature = torch.log(counts.float()) / np.log(2000)  # Normalized sample size
        
        # Precision calibration
        calibration_input = torch.cat([
            raw_precision, 
            sample_size_feature.unsqueeze(-1)
        ], dim=-1)
        
        calibrated_precision = self.precision_calibrator(calibration_input)
        calibrated_precision = torch.sigmoid(calibrated_precision)  # Ensure [0,1] range
        
        return tp_out, fp_out, ratio_out, calibrated_precision

def precision_calibrated_loss(predictions, targets, density_weight=1.0, ratio_weight=1.0, precision_weight=2.0):
    """
    Loss function that includes precision calibration
    """
    tp_pred, fp_pred, ratio_pred, calibrated_precision = predictions
    tp_target, fp_target, ratio_target, precision_target = targets
    
    # Standard losses for densities and ratio
    tp_loss = F.kl_div(torch.log(tp_pred + 1e-9), tp_target, reduction='batchmean')
    fp_loss = F.kl_div(torch.log(fp_pred + 1e-9), fp_target, reduction='batchmean')
    ratio_loss = F.mse_loss(ratio_pred, ratio_target)
    
    # NEW: Precision calibration loss
    precision_loss = F.mse_loss(calibrated_precision, precision_target)
    
    total_loss = (density_weight * (tp_loss + fp_loss) + 
                  ratio_weight * ratio_loss + 
                  precision_weight * precision_loss)
    
    return total_loss, tp_loss, fp_loss, ratio_loss, precision_loss

def compute_ground_truth_precision_per_bin(alerts_df, n_bins=20):
    """Compute ground truth precision for each probability bin"""
    bins = np.linspace(0, 1, n_bins + 1)
    precision_per_bin = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (alerts_df['max_proba'] >= bins[i]) & (alerts_df['max_proba'] < bins[i+1])
        bin_data = alerts_df[mask]
        
        if len(bin_data) > 0:
            tp_count = len(bin_data[bin_data['is_theft'] == 1])
            precision_per_bin[i] = tp_count / len(bin_data)
        else:
            precision_per_bin[i] = 0.0  # Empty bin
    
    return precision_per_bin

def prepare_calibrated_training_sample(camera_data, sample_size_range=(50, 2000)):
    """Prepare a training sample with ground truth precision per bin"""
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
    
    # Compute TP ratio from actual alert counts
    tp_count = len(alerts_df[alerts_df['is_theft'] == 1])
    total_count = len(alerts_df)
    tp_ratio = tp_count / total_count if total_count > 0 else 0.0
    
    # NEW: Compute ground truth precision per bin from FULL camera data
    gt_precision_per_bin = compute_ground_truth_precision_per_bin(alerts_df)
    
    return {
        'features': torch.FloatTensor(features),
        'tp_histogram': torch.FloatTensor(tp_density),
        'fp_histogram': torch.FloatTensor(fp_density),
        'tp_ratio': tp_ratio,
        'precision_target': torch.FloatTensor(gt_precision_per_bin),
        'count': k
    }

def train_precision_calibrated_model(trial=None):
    """Train the precision-calibrated model"""
    
    # Hyperparameters
    if trial:
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [2, 4, 8])
        phi_dim = trial.suggest_categorical('phi_dim', [128, 208, 256])
        num_heads = trial.suggest_categorical('num_heads', [4, 7, 8])
        epochs = trial.suggest_int('epochs', 15, 30)
        precision_weight = trial.suggest_float('precision_weight', 1.0, 5.0)
    else:
        # Use best hyperparameters from previous training
        learning_rate = 0.0005793892334263356
        batch_size = 4
        phi_dim = 208
        num_heads = 7
        epochs = 25
        precision_weight = 3.0  # Higher weight on precision calibration
    
    # Load training data
    print("🔄 Loading and preparing training data...")
    with open("ground_truth_histograms.pkl", 'rb') as f:
        data = pickle.load(f)
    
    # Split into train/val
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    print(f"🔧 Training precision-calibrated model...")
    print(f"   Precision weight: {precision_weight}x")
    print(f"   Batch size: {batch_size}, Learning rate: {learning_rate:.6f}")
    print(f"   Training cameras: {len(train_data)}, Validation: {len(val_data)}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepSetsPrecisionCalibrated(phi_dim, 20, num_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_losses = {'total': [], 'tp': [], 'fp': [], 'ratio': [], 'precision': []}
        
        # Generate training batches on the fly
        num_batches = len(train_data) // batch_size
        
        for batch_idx in range(num_batches):
            # Sample cameras for this batch
            batch_cameras = np.random.choice(train_data, size=batch_size, replace=False)
            
            # Prepare batch
            batch_samples = []
            for camera in batch_cameras:
                sample = prepare_calibrated_training_sample(camera)
                if sample is not None:
                    batch_samples.append(sample)
            
            if len(batch_samples) == 0:
                continue
                
            # Convert to tensors
            features_batch = torch.stack([s['features'] for s in batch_samples]).to(device)
            tp_targets = torch.stack([s['tp_histogram'] for s in batch_samples]).to(device)
            fp_targets = torch.stack([s['fp_histogram'] for s in batch_samples]).to(device)
            ratio_targets = torch.tensor([s['tp_ratio'] for s in batch_samples], dtype=torch.float32).to(device)
            precision_targets = torch.stack([s['precision_target'] for s in batch_samples]).to(device)
            counts = torch.tensor([s['count'] for s in batch_samples], dtype=torch.long).to(device)
            
            # Forward pass
            tp_pred, fp_pred, ratio_pred, calibrated_precision = model(features_batch, counts)
            
            # Calculate precision-calibrated loss
            predictions = (tp_pred, fp_pred, ratio_pred, calibrated_precision)
            targets = (tp_targets, fp_targets, ratio_targets, precision_targets)
            total_loss, tp_loss, fp_loss, ratio_loss, precision_loss = precision_calibrated_loss(
                predictions, targets, precision_weight=precision_weight
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track losses
            epoch_losses['total'].append(total_loss.item())
            epoch_losses['tp'].append(tp_loss.item())
            epoch_losses['fp'].append(fp_loss.item())
            epoch_losses['ratio'].append(ratio_loss.item())
            epoch_losses['precision'].append(precision_loss.item())
        
        # Print progress
        if epoch % 5 == 0 or epoch == epochs - 1:
            avg_total = np.mean(epoch_losses['total'])
            avg_tp = np.mean(epoch_losses['tp'])
            avg_fp = np.mean(epoch_losses['fp'])
            avg_ratio = np.mean(epoch_losses['ratio'])
            avg_precision = np.mean(epoch_losses['precision'])
            
            print(f"Epoch {epoch+1:2d}: Total={avg_total:.4f} "
                  f"(TP={avg_tp:.4f}, FP={avg_fp:.4f}, Ratio={avg_ratio:.4f}, Precision={avg_precision:.4f})")
    
    # Evaluate model
    print("🧪 Evaluating model...")
    r2_score, r2_tp, r2_fp, r2_ratio, r2_precision = evaluate_calibrated_model(model, val_data[:50])
    
    print(f"✅ Training complete!")
    print(f"   Overall R²: {r2_score:.4f}")
    print(f"   TP R²: {r2_tp:.4f}, FP R²: {r2_fp:.4f}, Ratio R²: {r2_ratio:.4f}")
    print(f"   🎯 Precision R²: {r2_precision:.4f}")
    
    # Save model
    if not trial:  # Only save when not in hyperparameter search
        save_path = "runs/best_model/best_checkpoint_precision_calibrated.pth"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'r2_score': r2_score,
            'r2_tp': r2_tp,
            'r2_fp': r2_fp,
            'r2_ratio': r2_ratio,
            'r2_precision': r2_precision,
            'hyperparameters': {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'phi_dim': phi_dim,
                'num_heads': num_heads,
                'epochs': epochs,
                'precision_weight': precision_weight
            }
        }, save_path)
        print(f"💾 Model saved to {save_path}")
    
    return r2_score

def evaluate_calibrated_model(model, test_data):
    """Evaluate precision-calibrated model performance"""
    model.eval()
    device = next(model.parameters()).device
    
    tp_predictions = []
    fp_predictions = []
    ratio_predictions = []
    precision_predictions = []
    tp_targets = []
    fp_targets = []
    ratio_targets = []
    precision_targets = []
    
    with torch.no_grad():
        for camera_data in test_data:
            sample = prepare_calibrated_training_sample(camera_data, sample_size_range=(100, 1000))
            if sample is None:
                continue
                
            features = sample['features'].unsqueeze(0).to(device)
            count = torch.tensor([sample['count']], dtype=torch.long).to(device)
            
            tp_pred, fp_pred, ratio_pred, precision_pred = model(features, count)
            
            tp_predictions.append(tp_pred.cpu().numpy()[0])
            fp_predictions.append(fp_pred.cpu().numpy()[0])
            ratio_predictions.append(ratio_pred.cpu().item())
            precision_predictions.append(precision_pred.cpu().numpy()[0])
            
            # Use the original camera data for targets
            tp_targets.append(camera_data['tp_density'])
            fp_targets.append(camera_data['fp_density'])
            
            # Compute ground truth ratio and precision
            alerts_df = pd.read_parquet(camera_data['file_path'])
            tp_count = len(alerts_df[alerts_df['is_theft'] == 1])
            total_count = len(alerts_df)
            gt_ratio = tp_count / total_count if total_count > 0 else 0.0
            ratio_targets.append(gt_ratio)
            
            gt_precision_per_bin = compute_ground_truth_precision_per_bin(alerts_df)
            precision_targets.append(gt_precision_per_bin)
    
    if len(tp_predictions) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Calculate R² scores
    from sklearn.metrics import r2_score
    
    tp_predictions = np.array(tp_predictions)
    fp_predictions = np.array(fp_predictions)
    ratio_predictions = np.array(ratio_predictions)
    precision_predictions = np.array(precision_predictions)
    tp_targets = np.array(tp_targets)
    fp_targets = np.array(fp_targets)
    ratio_targets = np.array(ratio_targets)
    precision_targets = np.array(precision_targets)
    
    # Flatten for R² calculation
    tp_flat_pred = tp_predictions.flatten()
    tp_flat_target = tp_targets.flatten()
    fp_flat_pred = fp_predictions.flatten()
    fp_flat_target = fp_targets.flatten()
    precision_flat_pred = precision_predictions.flatten()
    precision_flat_target = precision_targets.flatten()
    
    r2_tp = r2_score(tp_flat_target, tp_flat_pred)
    r2_fp = r2_score(fp_flat_target, fp_flat_pred)
    r2_ratio = r2_score(ratio_targets, ratio_predictions)
    r2_precision = r2_score(precision_flat_target, precision_flat_pred)
    
    # Overall R² (weighted by importance, with higher weight on precision)
    r2_overall = (r2_tp + r2_fp + r2_ratio + 2*r2_precision) / 5
    
    return r2_overall, r2_tp, r2_fp, r2_ratio, r2_precision

if __name__ == "__main__":
    print("🚀 PRECISION-CALIBRATED TRAINING")
    print("=" * 50)
    print("Strategy: Keep density fitting + Add precision calibration layer")
    print("Hypothesis: Direct precision correction will fix systematic bias")
    print()
    
    # Train the precision-calibrated model
    final_score = train_precision_calibrated_model()
    
    print(f"\n🎯 Final R² Score: {final_score:.4f}")
    print("🔍 Next: Test precision calibration against baseline!") 