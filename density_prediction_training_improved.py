#!/usr/bin/env python3
"""
Improved Density Prediction Model Training Pipeline
Handles full range of camera sample sizes (10-2000 alerts)
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
from typing import List, Tuple, Dict
import warnings
from tqdm import tqdm
import os
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import argparse
from torch.distributions import Dirichlet

warnings.filterwarnings('ignore')

N_BINS = 20
SAMPLE_SIZE_RANGE = (10, 2000)  # Cover full range of camera sizes
DEFAULT_BATCH_SIZE = 8  # Smaller batch for larger sequences
DEFAULT_EPOCHS = 40
DEFAULT_LEARNING_RATE = 5e-6  # Lower learning rate for stability
DEFAULT_PHI_DIM = 128
DEFAULT_NUM_HEADS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CameraDensityDatasetImproved(Dataset):
    def __init__(self, processed_data: List[Dict], sample_size_range: Tuple[int, int]):
        self.sample_size_range = sample_size_range
        self.camera_data = processed_data
        self.df_cache = {cam['file_path']: pd.read_parquet(cam['file_path']) 
                        for cam in tqdm(processed_data, desc="Caching DataFrames")}
        
        # Analyze camera size distribution for smarter sampling
        self.camera_sizes = [len(df) for df in self.df_cache.values()]
        self.size_percentiles = np.percentile(self.camera_sizes, [25, 50, 75, 90, 95])
        print(f"Camera size percentiles: {self.size_percentiles}")

    def __len__(self):
        return len(self.camera_data)

    def smart_sample_size(self, camera_size: int) -> int:
        """Smart sampling strategy that covers realistic ranges"""
        min_size, max_size = self.sample_size_range
        
        # Ensure we don't exceed camera size or max training size
        effective_max = min(max_size, camera_size)
        
        if effective_max <= min_size:
            return min_size
        
        # Define sampling strategies with proper bounds checking
        def safe_small_sample():
            upper = min(100, camera_size, max_size)
            return random.randint(min_size, upper)
        
        def safe_medium_sample():
            lower = max(min_size, int(0.05 * camera_size))
            upper = min(max_size, int(0.25 * camera_size), camera_size)
            if upper <= lower:
                return lower
            return random.randint(lower, upper)
        
        def safe_large_sample():
            lower = max(min_size, int(0.25 * camera_size))
            upper = min(max_size, int(0.8 * camera_size), camera_size)
            if upper <= lower:
                return min(max_size, camera_size)
            return random.randint(lower, upper)
        
        def safe_uniform_sample():
            return random.randint(min_size, effective_max)
        
        strategies = [safe_small_sample, safe_medium_sample, safe_large_sample, safe_uniform_sample]
        
        # Choose strategy based on camera size and random chance
        if camera_size < 50:
            # Small cameras: mostly use small samples
            weights = [0.8, 0.2, 0.0, 0.0]
        elif camera_size < 200:
            # Medium cameras: balanced approach
            weights = [0.4, 0.4, 0.1, 0.1]
        else:
            # Large cameras: cover full range
            weights = [0.2, 0.3, 0.3, 0.2]
        
        strategy = np.random.choice(strategies, p=weights)
        return strategy()

    def __getitem__(self, idx):
        camera = self.camera_data[idx]
        df = self.df_cache[camera['file_path']]
        
        # Smart sampling for better coverage
        k = self.smart_sample_size(len(df))
        sample_df = df.sample(n=k, replace=False)
        
        base_features = torch.tensor(sample_df[['max_proba', 'is_theft']].values, dtype=torch.float32)
        
        # Multi-scale size features for better generalization
        max_size = self.sample_size_range[1]
        
        # Linear normalization (0-1)
        k_linear = k / max_size
        # Log normalization (0-1)
        k_log = np.log(k) / np.log(max_size)
        # Percentile-based normalization
        k_percentile = np.searchsorted(self.size_percentiles, k) / len(self.size_percentiles)
        
        # Combine multiple size features
        size_features = torch.tensor([k_linear, k_log, k_percentile], dtype=torch.float32)
        k_feature = size_features.unsqueeze(0).repeat(k, 1)
        
        sample_features = torch.cat([base_features, k_feature], dim=1)
        
        tp_density = torch.tensor(camera['tp_density'], dtype=torch.float32)
        fp_density = torch.tensor(camera['fp_density'], dtype=torch.float32)
        return sample_features, tp_density, fp_density, k

class ResidualMLP(nn.Module):
    def __init__(self, dim):
        super(ResidualMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )
    def forward(self, x): return x + self.net(x)

class MAB(nn.Module):
    def __init__(self, dim, num_heads):
        super(MAB, self).__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    def forward(self, Q, K): 
        attn_out, _ = self.attn(Q, K, K)
        return self.norm(Q + attn_out)

class DeepSetsAdvancedImproved(nn.Module):
    def __init__(self, phi_dim, n_bins, num_heads):
        super(DeepSetsAdvancedImproved, self).__init__()
        if phi_dim % num_heads != 0: 
            phi_dim = (phi_dim // num_heads + 1) * num_heads
        
        # Input has 5 features: max_proba, is_theft, k_linear, k_log, k_percentile
        self.phi = nn.Sequential(
            nn.Linear(5, phi_dim), 
            ResidualMLP(phi_dim), 
            ResidualMLP(phi_dim)
        )
        
        # Multi-head attention for pooling
        self.pooling = MAB(phi_dim, num_heads)
        self.query = nn.Parameter(torch.randn(1, 1, phi_dim))
        
        # Enhanced processing
        self.rho = nn.Sequential(
            ResidualMLP(phi_dim), 
            ResidualMLP(phi_dim),
            ResidualMLP(phi_dim)  # Extra layer for complexity
        )
        
        # Output heads
        self.tp_head = nn.Linear(phi_dim, n_bins)
        self.fp_head = nn.Linear(phi_dim, n_bins)
        
    def forward(self, x, counts):
        # Create mask for variable-length sequences
        mask = torch.arange(x.size(1), device=x.device)[None, :] < counts[:, None]
        
        # Apply phi to all elements
        phi_out = self.phi(x * mask.unsqueeze(-1))
        
        # Attention-based pooling
        query = self.query.repeat(x.shape[0], 1, 1)
        agg = self.pooling(query, phi_out).squeeze(1)
        
        # Process aggregated representation
        rho_out = self.rho(agg)
        
        # Generate normalized density predictions
        tp_out = F.softmax(self.tp_head(rho_out), dim=1)
        fp_out = F.softmax(self.fp_head(rho_out), dim=1)
        
        return tp_out, fp_out

def mse_loss(pred, target): 
    return F.mse_loss(pred, target)

def collate_fn_improved(batch):
    features, tps, fps, counts = zip(*[b for b in batch if b is not None])
    if not features: 
        return None, None, None, None
    
    # Handle variable sequence lengths more efficiently
    max_len = max(len(f) for f in features)
    batch_size = len(features)
    
    # Pre-allocate with correct feature size (5 features now)
    padded = torch.zeros(batch_size, max_len, 5)
    
    for i, f in enumerate(features): 
        padded[i, :len(f), :] = f
        
    return padded, torch.stack(tps), torch.stack(fps), torch.tensor(counts, dtype=torch.float32)

def plot_densities_improved(epoch, plot_data, n_bins, out_dir):
    """Enhanced plotting with size information"""
    fig, axes = plt.subplots(5, 2, figsize=(20, 24))
    fig.suptitle(f'Epoch {epoch} - Improved Training (10-2000 alerts)', fontsize=20)
    
    x = np.linspace(0, 1, n_bins)
    width = x[1] - x[0]
    
    for i, data in enumerate(plot_data):
        # TP Plot
        ax_tp = axes[i, 0]
        gt_tp_mean = data['gt_tp'] / (data['gt_tp'].sum() + 1e-9)
        pred_tp_mean = data['pred_tp'] / (data['pred_tp'].sum() + 1e-9)
        
        ax_tp.bar(x, data['tp_subset_hist'], width=width, label='TP Subset Hist', color='green', alpha=0.5)
        ax_tp.plot(x, gt_tp_mean, label='GT Mean', color='blue', linewidth=2)
        ax_tp.plot(x, pred_tp_mean, label='Pred Mean', color='red', linestyle='--', linewidth=2)
        ax_tp.set_title(f"Cam {i+1} TP (k={data['k']} alerts)")
        ax_tp.legend()
        ax_tp.grid(True, alpha=0.3)

        # FP Plot
        ax_fp = axes[i, 1]
        gt_fp_mean = data['gt_fp'] / (data['gt_fp'].sum() + 1e-9)
        pred_fp_mean = data['pred_fp'] / (data['pred_fp'].sum() + 1e-9)
        
        ax_fp.bar(x, data['fp_subset_hist'], width=width, label='FP Subset Hist', color='green', alpha=0.5)
        ax_fp.plot(x, gt_fp_mean, label='GT Mean', color='blue', linewidth=2)
        ax_fp.plot(x, pred_fp_mean, label='Pred Mean', color='red', linestyle='--', linewidth=2)
        ax_fp.set_title(f"Cam {i+1} FP (k={data['k']} alerts)")
        ax_fp.legend()
        ax_fp.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(out_dir, f"epoch_{epoch}_improved.png"), dpi=200)
    plt.close(fig)

def main(args):
    with open("ground_truth_histograms.pkl", 'rb') as f: 
        data = pickle.load(f)
    
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    train_ds = CameraDensityDatasetImproved(train_data, SAMPLE_SIZE_RANGE)
    val_ds = CameraDensityDatasetImproved(val_data, SAMPLE_SIZE_RANGE)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                             num_workers=4, collate_fn=collate_fn_improved)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                           num_workers=4, collate_fn=collate_fn_improved)
    
    model = DeepSetsAdvancedImproved(args.phi_dim, N_BINS, args.num_heads).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.8)
    
    plots_dir = "plots_improved"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Initialize best model tracking
    best_r2_score = -1.0
    best_model_path = "runs/best_model/best_checkpoint_improved.pth"
    
    print(f"Starting improved training on {DEVICE}...")
    print(f"Training range: {SAMPLE_SIZE_RANGE[0]}-{SAMPLE_SIZE_RANGE[1]} alerts")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            if batch[0] is None: continue
            features, gt_tp, gt_fp, counts = [b.to(DEVICE) for b in batch]
            
            optimizer.zero_grad()
            pred_tp, pred_fp = model(features, counts)
            loss = mse_loss(pred_tp, gt_tp) + mse_loss(pred_fp, gt_fp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        plot_data = []
        epoch_gt_tp_mean, epoch_pred_tp_mean = [], []
        epoch_gt_fp_mean, epoch_pred_fp_mean = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                if batch[0] is None: continue
                features, gt_tp, gt_fp, counts = [b.to(DEVICE) for b in batch]
                pred_tp, pred_fp = model(features, counts)
                val_loss = mse_loss(pred_tp, gt_tp) + mse_loss(pred_fp, gt_fp)
                total_val_loss += val_loss.item()
                
                # Collect for R² calculation
                epoch_gt_tp_mean.append(gt_tp.cpu().numpy())
                epoch_pred_tp_mean.append(pred_tp.cpu().numpy())
                epoch_gt_fp_mean.append(gt_fp.cpu().numpy())
                epoch_pred_fp_mean.append(pred_fp.cpu().numpy())
            
            # Generate plots for first 5 validation samples
            for i in range(min(5, len(val_ds))):
                features, gt_tp, gt_fp, k = val_ds[i]
                pred_tp, pred_fp = model(features.unsqueeze(0).to(DEVICE), torch.tensor([k]).to(DEVICE))
                
                # Calculate subset histograms
                tp_subset_probs = features[features[:, 1] == 1][:, 0].numpy()
                fp_subset_probs = features[features[:, 1] == 0][:, 0].numpy()

                tp_hist, _ = np.histogram(tp_subset_probs, bins=N_BINS, range=(0,1))
                fp_hist, _ = np.histogram(fp_subset_probs, bins=N_BINS, range=(0,1))

                plot_data.append({
                    'k': k,
                    'gt_tp': gt_tp.numpy(),
                    'pred_tp': pred_tp.squeeze().cpu().numpy(),
                    'gt_fp': gt_fp.numpy(),
                    'pred_fp': pred_fp.squeeze().cpu().numpy(),
                    'tp_subset_hist': tp_hist / (tp_hist.sum() + 1e-9),
                    'fp_subset_hist': fp_hist / (fp_hist.sum() + 1e-9)
                })
                
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        # Calculate R² score
        if epoch_gt_tp_mean:
            epoch_gt_tp_mean = np.concatenate(epoch_gt_tp_mean)
            epoch_pred_tp_mean = np.concatenate(epoch_pred_tp_mean)
            epoch_gt_fp_mean = np.concatenate(epoch_gt_fp_mean)
            epoch_pred_fp_mean = np.concatenate(epoch_pred_fp_mean)

            from sklearn.metrics import r2_score
            try:
                epoch_r2_tp = r2_score(epoch_gt_tp_mean.T, epoch_pred_tp_mean.T)
                epoch_r2_fp = r2_score(epoch_gt_fp_mean.T, epoch_pred_fp_mean.T)
                epoch_avg_r2 = (epoch_r2_tp + epoch_r2_fp) / 2
            except Exception as e:
                print(f"R² calculation failed: {e}")
                epoch_r2_tp = epoch_r2_fp = epoch_avg_r2 = 0.0
            
            # Save best model
            if epoch_avg_r2 > best_r2_score:
                best_r2_score = epoch_avg_r2
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'r2_score': epoch_avg_r2,
                    'r2_tp': epoch_r2_tp,
                    'r2_fp': epoch_r2_fp,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'hyperparameters': {
                        'learning_rate': args.learning_rate,
                        'batch_size': args.batch_size,
                        'phi_dim': args.phi_dim,
                        'num_heads': args.num_heads,
                        'epochs': args.epochs,
                        'sample_range': SAMPLE_SIZE_RANGE
                    }
                }, best_model_path)
                
                plot_densities_improved(epoch, plot_data, N_BINS, plots_dir)
                print(f"Epoch {epoch} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                      f"R²: {epoch_avg_r2:.4f} (TP: {epoch_r2_tp:.4f}, FP: {epoch_r2_fp:.4f}) | "
                      f"✅ BEST MODEL SAVED")
            else:
                if epoch % 5 == 0:  # Plot every 5 epochs
                    plot_densities_improved(epoch, plot_data, N_BINS, plots_dir)
                print(f"Epoch {epoch} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                      f"R²: {epoch_avg_r2:.4f} (TP: {epoch_r2_tp:.4f}, FP: {epoch_r2_fp:.4f})")
        else:
            print(f"Epoch {epoch} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

    print(f"\nTraining complete! Best R² score: {best_r2_score:.4f}")
    print(f"Best model saved to: {best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--phi_dim', type=int, default=DEFAULT_PHI_DIM)
    parser.add_argument('--num_heads', type=int, default=DEFAULT_NUM_HEADS)
    main(parser.parse_args()) 