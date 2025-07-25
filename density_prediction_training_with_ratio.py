#!/usr/bin/env python3
"""
Density + TP Ratio Prediction Model Training Pipeline
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

warnings.filterwarnings('ignore')

N_BINS = 20
SAMPLE_SIZE_RANGE = (10, 2000)  # Cover 95th percentile of camera sizes
DEFAULT_BATCH_SIZE = 16  # Reduced batch size for larger sequences
DEFAULT_EPOCHS = 30  # More epochs for complex task
DEFAULT_LEARNING_RATE = 0.0000731 # Reduced by a factor of 10
DEFAULT_PHI_DIM = 166
DEFAULT_NUM_HEADS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CameraDensityDatasetWithRatio(Dataset):
    def __init__(self, processed_data: List[Dict], sample_size_range: Tuple[int, int]):
        self.sample_size_range = sample_size_range
        self.camera_data = processed_data
        self.df_cache = {}
        
        # Cache DataFrames and calculate TP ratios
        print("Caching DataFrames and calculating TP ratios...")
        for cam in tqdm(processed_data, desc="Processing cameras"):
            df = pd.read_parquet(cam['file_path'])
            self.df_cache[cam['file_path']] = df
            
            # Calculate TP ratio for this camera
            total_alerts = len(df)
            tp_alerts = len(df[df['is_theft'] == 1])
            tp_ratio = tp_alerts / total_alerts if total_alerts > 0 else 0.0
            cam['tp_ratio'] = tp_ratio

    def __len__(self):
        return len(self.camera_data)

    def __getitem__(self, idx):
        camera = self.camera_data[idx]
        df = self.df_cache[camera['file_path']]
        min_size, max_size = self.sample_size_range
        
        # Smart sampling: prefer common sizes but cover full range
        camera_size = len(df)
        if random.random() < 0.7:  # 70% of time: sample from realistic range
            # Sample from 1% to 80% of camera size, but cap at max_size
            max_pct = min(0.8, max_size / camera_size) if camera_size > 0 else 0.8
            pct = random.uniform(0.01, max_pct)
            k = max(min_size, int(camera_size * pct))
        else:  # 30% of time: sample from full training range
            k = random.randint(min_size, min(max_size, camera_size))
        
        k = min(k, camera_size)  # Ensure we don't exceed camera size
        sample_df = df.sample(n=k, replace=False)
        base_features = torch.tensor(sample_df[['max_proba', 'is_theft']].values, dtype=torch.float32)
        
        # Improved normalization: use log scale for better handling of large sizes
        k_normalized = np.log(k) / np.log(max_size)  # Log-scale normalization
        k_feature = torch.full((k, 1), fill_value=k_normalized, dtype=torch.float32)
        sample_features = torch.cat([base_features, k_feature], dim=1)
        
        tp_density = torch.tensor(camera['tp_density'], dtype=torch.float32)
        fp_density = torch.tensor(camera['fp_density'], dtype=torch.float32)
        tp_ratio = torch.tensor(camera['tp_ratio'], dtype=torch.float32)
        
        return sample_features, tp_density, fp_density, tp_ratio, k

class ResidualMLP(nn.Module):
    def __init__(self, dim): 
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
    
    def forward(self, x): 
        return F.relu(self.fc2(F.relu(self.fc1(x))) + x)

class MAB(nn.Module):
    def __init__(self, dim_V, num_heads, ln=True):
        super(MAB, self).__init__()
        self.mha = nn.MultiheadAttention(dim_V, num_heads)
        self.ln1 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.ln2 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.ffn = nn.Sequential(nn.Linear(dim_V, dim_V), nn.ReLU(), nn.Linear(dim_V, dim_V))
    
    def forward(self, Q, K):
        Q_norm, K_norm = self.ln1(Q).permute(1, 0, 2), self.ln1(K).permute(1, 0, 2)
        out, _ = self.mha(Q_norm, K_norm, K_norm)
        out = Q + out.permute(1, 0, 2)
        out = self.ln2(out)
        out = out + self.ffn(out)
        return out

class DeepSetsAdvancedWithRatio(nn.Module):
    def __init__(self, phi_dim, n_bins, num_heads):
        super(DeepSetsAdvancedWithRatio, self).__init__()
        if phi_dim % num_heads != 0: 
            phi_dim = (phi_dim // num_heads + 1) * num_heads
        
        self.phi = nn.Sequential(
            nn.Linear(3, phi_dim), 
            ResidualMLP(phi_dim), 
            ResidualMLP(phi_dim)
        )
        self.pooling = MAB(phi_dim, num_heads)
        self.query = nn.Parameter(torch.randn(1, 1, phi_dim))
        self.rho = nn.Sequential(ResidualMLP(phi_dim), ResidualMLP(phi_dim))
        
        # Three output heads: TP density, FP density, and TP ratio
        self.tp_head = nn.Linear(phi_dim, n_bins)
        self.fp_head = nn.Linear(phi_dim, n_bins)
        self.ratio_head = nn.Linear(phi_dim, 1)  # Single output for TP ratio
    
    def forward(self, x, counts):
        mask = torch.arange(x.size(1), device=x.device)[None, :] < counts[:, None]
        phi_out = self.phi(x * mask.unsqueeze(-1))
        query = self.query.repeat(x.shape[0], 1, 1)
        agg = self.pooling(query, phi_out).squeeze(1)
        rho_out = self.rho(agg)
        
        tp_out = F.softmax(self.tp_head(rho_out), dim=1)
        fp_out = F.softmax(self.fp_head(rho_out), dim=1)
        ratio_out = torch.sigmoid(self.ratio_head(rho_out)).squeeze(-1)  # Sigmoid for [0,1] range
        
        return tp_out, fp_out, ratio_out

def combined_loss(pred_tp, pred_fp, pred_ratio, target_tp, target_fp, target_ratio, ratio_weight=1.0):
    """Combined loss for densities and TP ratio"""
    density_loss = F.mse_loss(pred_tp, target_tp) + F.mse_loss(pred_fp, target_fp)
    ratio_loss = F.mse_loss(pred_ratio, target_ratio)
    return density_loss + ratio_weight * ratio_loss, density_loss, ratio_loss

def collate_fn(batch):
    features, tps, fps, ratios, counts = zip(*[b for b in batch if b is not None])
    if not features: 
        return None, None, None, None, None
    
    max_len = max(len(f) for f in features)
    padded = torch.zeros(len(features), max_len, 3)
    for i, f in enumerate(features): 
        padded[i, :len(f), :] = f
    
    return (padded, torch.stack(tps), torch.stack(fps), 
            torch.stack(ratios), torch.tensor(counts, dtype=torch.float32))

def plot_densities_with_ratio(epoch, plot_data, n_bins, out_dir):
    fig, axes = plt.subplots(5, 3, figsize=(24, 22))  # Added third column for ratio
    fig.suptitle(f'Epoch {epoch}', fontsize=20)
    x = np.linspace(0, 1, n_bins)
    width = x[1] - x[0]
    
    for i, data in enumerate(plot_data):
        # TP Plot
        ax_tp = axes[i, 0]
        gt_tp_mean = data['gt_tp'] / (data['gt_tp'].sum() + 1e-9)
        pred_tp_mean = data['pred_tp'] / (data['pred_tp'].sum() + 1e-9)
        ax_tp.bar(x, data['tp_subset_hist'], width=width, label='TP Subset Hist', color='green', alpha=0.5)
        ax_tp.plot(x, gt_tp_mean, label='GT Mean', color='blue')
        ax_tp.plot(x, pred_tp_mean, label='Pred Mean', color='red', linestyle='--')
        ax_tp.set_title(f"Cam {i+1} TP (k={data['k']})")
        ax_tp.legend()

        # FP Plot
        ax_fp = axes[i, 1]
        gt_fp_mean = data['gt_fp'] / (data['gt_fp'].sum() + 1e-9)
        pred_fp_mean = data['pred_fp'] / (data['pred_fp'].sum() + 1e-9)
        ax_fp.bar(x, data['fp_subset_hist'], width=width, label='FP Subset Hist', color='green', alpha=0.5)
        ax_fp.plot(x, gt_fp_mean, label='GT Mean', color='blue')
        ax_fp.plot(x, pred_fp_mean, label='Pred Mean', color='red', linestyle='--')
        ax_fp.set_title(f"Cam {i+1} FP (k={data['k']})")
        ax_fp.legend()
        
        # TP Ratio Plot
        ax_ratio = axes[i, 2]
        gt_ratio = data['gt_ratio']
        pred_ratio = data['pred_ratio']
        subset_ratio = data['subset_ratio']
        
        # Bar chart showing the three ratios
        categories = ['Subset', 'Ground Truth', 'Predicted']
        values = [subset_ratio, gt_ratio, pred_ratio]
        colors = ['green', 'blue', 'red']
        bars = ax_ratio.bar(categories, values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax_ratio.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax_ratio.set_title(f"Cam {i+1} TP Ratio")
        ax_ratio.set_ylabel('TP Ratio')
        ax_ratio.set_ylim(0, 1)
        ax_ratio.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(out_dir, f"epoch_{epoch}.png"))
    plt.close(fig)

def main(args):
    with open("ground_truth_histograms.pkl", 'rb') as f: 
        data = pickle.load(f)
    
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    train_ds = CameraDensityDatasetWithRatio(train_data, SAMPLE_SIZE_RANGE)
    val_ds = CameraDensityDatasetWithRatio(val_data, SAMPLE_SIZE_RANGE)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                             num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                           num_workers=4, collate_fn=collate_fn)
    
    model = DeepSetsAdvancedWithRatio(args.phi_dim, N_BINS, args.num_heads).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    plots_dir = "plots_with_ratio"
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Starting training on {DEVICE}...")
    
    # Initialize best score tracking
    best_r2_score = -1.0
    best_model_path = "runs/best_model/best_checkpoint_with_ratio.pth"
    os.makedirs("runs/best_model", exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0
        total_train_density_loss = 0
        total_train_ratio_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            if batch[0] is None: 
                continue
            
            features, gt_tp, gt_fp, gt_ratio, counts = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            pred_tp, pred_fp, pred_ratio = model(features, counts)
            
            total_loss, density_loss, ratio_loss = combined_loss(
                pred_tp, pred_fp, pred_ratio, gt_tp, gt_fp, gt_ratio
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient Clipping
            optimizer.step()
            
            total_train_loss += total_loss.item()
            total_train_density_loss += density_loss.item()
            total_train_ratio_loss += ratio_loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_density_loss = total_train_density_loss / len(train_loader)
        avg_train_ratio_loss = total_train_ratio_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        total_val_density_loss = 0
        total_val_ratio_loss = 0
        plot_data = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                if batch[0] is None: 
                    continue
                
                features, gt_tp, gt_fp, gt_ratio, counts = [b.to(DEVICE) for b in batch]
                pred_tp, pred_fp, pred_ratio = model(features, counts)
                
                total_loss, density_loss, ratio_loss = combined_loss(
                    pred_tp, pred_fp, pred_ratio, gt_tp, gt_fp, gt_ratio
                )
                
                total_val_loss += total_loss.item()
                total_val_density_loss += density_loss.item()
                total_val_ratio_loss += ratio_loss.item()
            
            # Prepare plot data
            for i in range(min(5, len(val_ds))):
                features, gt_tp, gt_fp, gt_ratio, k = val_ds[i]
                pred_tp, pred_fp, pred_ratio = model(
                    features.unsqueeze(0).to(DEVICE), 
                    torch.tensor([k]).to(DEVICE)
                )
                
                # Calculate subset TP ratio
                tp_count = (features[:, 1] == 1).sum().item()
                subset_ratio = tp_count / k if k > 0 else 0.0
                
                # Separate subset features into TP and FP for correct histogramming
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
                    'gt_ratio': gt_ratio.item(),
                    'pred_ratio': pred_ratio.squeeze().cpu().item(),
                    'subset_ratio': subset_ratio,
                    'tp_subset_hist': tp_hist / (tp_hist.sum() + 1e-9),
                    'fp_subset_hist': fp_hist / (fp_hist.sum() + 1e-9)
                })
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_density_loss = total_val_density_loss / len(val_loader)
        avg_val_ratio_loss = total_val_ratio_loss / len(val_loader)
        
        # Calculate R² score for this epoch
        epoch_gt_tp_mean, epoch_pred_tp_mean = [], []
        epoch_gt_fp_mean, epoch_pred_fp_mean = [], []
        epoch_gt_ratio, epoch_pred_ratio = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                if batch[0] is None: 
                    continue
                
                features, gt_tp, gt_fp, gt_ratio, counts = [b.to(DEVICE) for b in batch]
                pred_tp, pred_fp, pred_ratio = model(features, counts)

                # Ground truth and predictions are already normalized densities
                epoch_gt_tp_mean.append(gt_tp.cpu().numpy())
                epoch_pred_tp_mean.append(pred_tp.cpu().numpy())
                epoch_gt_fp_mean.append(gt_fp.cpu().numpy())
                epoch_pred_fp_mean.append(pred_fp.cpu().numpy())
                epoch_gt_ratio.append(gt_ratio.cpu().numpy())
                epoch_pred_ratio.append(pred_ratio.cpu().numpy())

        if epoch_gt_tp_mean:  # Only calculate if we have data
            epoch_gt_tp_mean = np.concatenate(epoch_gt_tp_mean)
            epoch_pred_tp_mean = np.concatenate(epoch_pred_tp_mean)
            epoch_gt_fp_mean = np.concatenate(epoch_gt_fp_mean)
            epoch_pred_fp_mean = np.concatenate(epoch_pred_fp_mean)
            epoch_gt_ratio = np.concatenate(epoch_gt_ratio)
            epoch_pred_ratio = np.concatenate(epoch_pred_ratio)

            from sklearn.metrics import r2_score
            try:
                # Check for NaN or invalid values
                if np.any(np.isnan(epoch_gt_tp_mean)) or np.any(np.isnan(epoch_pred_tp_mean)):
                    epoch_r2_tp = np.nan
                else:
                    epoch_r2_tp = r2_score(epoch_gt_tp_mean.T, epoch_pred_tp_mean.T)
                
                if np.any(np.isnan(epoch_gt_fp_mean)) or np.any(np.isnan(epoch_pred_fp_mean)):
                    epoch_r2_fp = np.nan
                else:
                    epoch_r2_fp = r2_score(epoch_gt_fp_mean.T, epoch_pred_fp_mean.T)
                
                if np.any(np.isnan(epoch_gt_ratio)) or np.any(np.isnan(epoch_pred_ratio)):
                    epoch_r2_ratio = np.nan
                else:
                    epoch_r2_ratio = r2_score(epoch_gt_ratio, epoch_pred_ratio)
                
                if np.isnan(epoch_r2_tp) or np.isnan(epoch_r2_fp):
                    epoch_avg_r2 = np.nan
                else:
                    epoch_avg_r2 = (epoch_r2_tp + epoch_r2_fp) / 2
                    
            except Exception as e:
                print(f"R² calculation failed: {e}")
                epoch_r2_tp = epoch_r2_fp = epoch_r2_ratio = epoch_avg_r2 = np.nan
            
            plot_densities_with_ratio(epoch, plot_data, N_BINS, plots_dir)
            r2_str = f"{epoch_avg_r2:.4f} (TP: {epoch_r2_tp:.4f}, FP: {epoch_r2_fp:.4f}, Ratio: {epoch_r2_ratio:.4f})" if not np.isnan(epoch_avg_r2) else "nan"
            
            # Save best model checkpoint
            if not np.isnan(epoch_avg_r2) and epoch_avg_r2 > best_r2_score:
                best_r2_score = epoch_avg_r2
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'r2_score': epoch_avg_r2,
                    'r2_tp': epoch_r2_tp,
                    'r2_fp': epoch_r2_fp,
                    'r2_ratio': epoch_r2_ratio,
                    'train_loss': avg_train_loss,
                    'train_density_loss': avg_train_density_loss,
                    'train_ratio_loss': avg_train_ratio_loss,
                    'val_loss': avg_val_loss,
                    'val_density_loss': avg_val_density_loss,
                    'val_ratio_loss': avg_val_ratio_loss,
                    'hyperparameters': {
                        'learning_rate': args.learning_rate,
                        'batch_size': args.batch_size,
                        'phi_dim': args.phi_dim,
                        'num_heads': args.num_heads,
                        'epochs': args.epochs
                    }
                }, best_model_path)
                print(f"Epoch {epoch} | Train: {avg_train_loss:.4f} (D:{avg_train_density_loss:.4f}, R:{avg_train_ratio_loss:.4f}) | Val: {avg_val_loss:.4f} (D:{avg_val_density_loss:.4f}, R:{avg_val_ratio_loss:.4f}) | R²: {r2_str} | ✅ BEST MODEL SAVED")
            else:
                print(f"Epoch {epoch} | Train: {avg_train_loss:.4f} (D:{avg_train_density_loss:.4f}, R:{avg_train_ratio_loss:.4f}) | Val: {avg_val_loss:.4f} (D:{avg_val_density_loss:.4f}, R:{avg_val_ratio_loss:.4f}) | R²: {r2_str}")
        else:
            plot_densities_with_ratio(epoch, plot_data, N_BINS, plots_dir)
            print(f"Epoch {epoch} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

    print("\nCalculating final R² score...")
    model.eval()
    all_gt_tp_mean, all_pred_tp_mean = [], []
    all_gt_fp_mean, all_pred_fp_mean = [], []
    all_gt_ratio, all_pred_ratio = [], []
    final_r2_score = -1.0  # Default to a bad score

    try:
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Final R² Calculation"):
                if batch[0] is None: 
                    continue
                
                features, gt_tp, gt_fp, gt_ratio, counts = [b.to(DEVICE) for b in batch]
                pred_tp, pred_fp, pred_ratio = model(features, counts)

                all_gt_tp_mean.append(gt_tp.cpu().numpy())
                all_pred_tp_mean.append(pred_tp.cpu().numpy())
                all_gt_fp_mean.append(gt_fp.cpu().numpy())
                all_pred_fp_mean.append(pred_fp.cpu().numpy())
                all_gt_ratio.append(gt_ratio.cpu().numpy())
                all_pred_ratio.append(pred_ratio.cpu().numpy())

        if all_gt_tp_mean:
            all_gt_tp_mean = np.concatenate(all_gt_tp_mean)
            all_pred_tp_mean = np.concatenate(all_pred_tp_mean)
            all_gt_fp_mean = np.concatenate(all_gt_fp_mean)
            all_pred_fp_mean = np.concatenate(all_pred_fp_mean)
            all_gt_ratio = np.concatenate(all_gt_ratio)
            all_pred_ratio = np.concatenate(all_pred_ratio)

            from sklearn.metrics import r2_score
            final_r2_tp = r2_score(all_gt_tp_mean.T, all_pred_tp_mean.T)
            final_r2_fp = r2_score(all_gt_fp_mean.T, all_pred_fp_mean.T)
            final_r2_ratio = r2_score(all_gt_ratio, all_pred_ratio)
            final_r2_score = (final_r2_tp + final_r2_fp) / 2
            
    except Exception as e:
        print(f"\nCould not calculate final R² score due to an error: {e}")
        final_r2_score = -1.0 # Ensure it reports a bad score on error

    print(f"Final R2 Score: {final_r2_score}")
    print(f"Final TP R2: {final_r2_tp if 'final_r2_tp' in locals() else -1}")
    print(f"Final FP R2: {final_r2_fp if 'final_r2_fp' in locals() else -1}")
    print(f"Final Ratio R2: {final_r2_ratio if 'final_r2_ratio' in locals() else -1}")
    
    # Save final model checkpoint
    final_model_path = "runs/best_model/final_checkpoint_with_ratio.pth"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_r2_score': final_r2_score,
        'final_r2_tp': final_r2_tp if 'final_r2_tp' in locals() else -1,
        'final_r2_fp': final_r2_fp if 'final_r2_fp' in locals() else -1,
        'final_r2_ratio': final_r2_ratio if 'final_r2_ratio' in locals() else -1,
        'hyperparameters': {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'phi_dim': args.phi_dim,
            'num_heads': args.num_heads,
            'epochs': args.epochs
        }
    }, final_model_path)
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {best_model_path} (R² = {best_r2_score:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--phi_dim', type=int, default=DEFAULT_PHI_DIM)
    parser.add_argument('--num_heads', type=int, default=DEFAULT_NUM_HEADS)
    main(parser.parse_args()) 