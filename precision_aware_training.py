#!/usr/bin/env python3
"""
Precision-Aware Training with Biased Ratio Loss
Implements a loss that optimizes for precision accuracy rather than ratio accuracy
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
SAMPLE_SIZE_RANGE = (10, 2000)
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 22
DEFAULT_LEARNING_RATE = 0.0005793892334263356
DEFAULT_PHI_DIM = 208
DEFAULT_NUM_HEADS = 7
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_WEIGHT_DECAY = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CameraDensityDatasetPrecisionAware(Dataset):
    def __init__(self, processed_data: List[Dict], sample_size_range: Tuple[int, int]):
        self.sample_size_range = sample_size_range
        self.camera_data = processed_data
        self.df_cache = {}
        
        # Cache DataFrames and calculate TP ratios + ground truth precision
        print("Caching DataFrames and calculating TP ratios + GT precision...")
        for cam in tqdm(processed_data, desc="Processing cameras"):
            df = pd.read_parquet(cam['file_path'])
            self.df_cache[cam['file_path']] = df
            
            # Calculate TP ratio
            total_alerts = len(df)
            tp_alerts = len(df[df['is_theft'] == 1])
            tp_ratio = tp_alerts / total_alerts if total_alerts > 0 else 0.0
            cam['tp_ratio'] = tp_ratio
            
            # Calculate ground truth TP ratio (simple precision)
            gt_precision = self.calculate_ground_truth_precision(df)
            cam['gt_precision'] = gt_precision

    def calculate_ground_truth_precision(self, camera_data):
        """Calculate ground truth TP ratio for a camera (same as what model should predict)"""
        tp_count = len(camera_data[camera_data['is_theft'] == 1])
        total_count = len(camera_data)
        
        if total_count == 0:
            return 0.0
        
        return tp_count / total_count

    def __len__(self):
        return len(self.camera_data)

    def __getitem__(self, idx):
        camera = self.camera_data[idx]
        df = self.df_cache[camera['file_path']]
        min_size, max_size = self.sample_size_range
        
        # Smart sampling
        camera_size = len(df)
        if random.random() < 0.7:
            max_pct = min(0.8, max_size / camera_size) if camera_size > 0 else 0.8
            pct = random.uniform(0.01, max_pct)
            k = max(min_size, int(camera_size * pct))
        else:
            k = random.randint(min_size, min(max_size, camera_size))
        
        k = min(k, camera_size)
        sample_df = df.sample(n=k, replace=False)
        base_features = torch.tensor(sample_df[['max_proba', 'is_theft']].values, dtype=torch.float32)
        
        # Log-scale normalization
        k_normalized = np.log(k) / np.log(max_size)
        k_feature = torch.full((k, 1), fill_value=k_normalized, dtype=torch.float32)
        sample_features = torch.cat([base_features, k_feature], dim=1)
        
        tp_density = torch.tensor(camera['tp_density'], dtype=torch.float32)
        fp_density = torch.tensor(camera['fp_density'], dtype=torch.float32)
        tp_ratio = torch.tensor(camera['tp_ratio'], dtype=torch.float32)
        gt_precision = torch.tensor(camera['gt_precision'], dtype=torch.float32)
        
        return sample_features, tp_density, fp_density, tp_ratio, gt_precision, k

class ResidualMLP(nn.Module):
    def __init__(self, dim, dropout_rate=0.1): 
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x): 
        return F.relu(self.fc2(self.dropout(F.relu(self.fc1(x))))) + x

class MAB(nn.Module):
    def __init__(self, dim_V, num_heads, dropout_rate=0.1, ln=True):
        super(MAB, self).__init__()
        self.mha = nn.MultiheadAttention(dim_V, num_heads, dropout=dropout_rate)
        self.ln1 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.ln2 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(dim_V, dim_V), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(dim_V, dim_V)
        )
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, Q, K):
        Q_norm, K_norm = self.ln1(Q).permute(1, 0, 2), self.ln1(K).permute(1, 0, 2)
        out, _ = self.mha(Q_norm, K_norm, K_norm)
        out = Q + self.dropout(out.permute(1, 0, 2))
        out = self.ln2(out)
        out = out + self.ffn(out)
        return out

class DeepSetsPrecisionAware(nn.Module):
    def __init__(self, phi_dim, n_bins, num_heads, dropout_rate=0.1):
        super(DeepSetsPrecisionAware, self).__init__()
        if phi_dim % num_heads != 0: 
            phi_dim = (phi_dim // num_heads + 1) * num_heads
        
        self.phi = nn.Sequential(
            nn.Linear(3, phi_dim),
            nn.Dropout(dropout_rate),
            ResidualMLP(phi_dim, dropout_rate), 
            ResidualMLP(phi_dim, dropout_rate)
        )
        self.pooling = MAB(phi_dim, num_heads, dropout_rate)
        self.query = nn.Parameter(torch.randn(1, 1, phi_dim))
        self.rho = nn.Sequential(
            ResidualMLP(phi_dim, dropout_rate), 
            ResidualMLP(phi_dim, dropout_rate)
        )
        
        # Three output heads: TP density, FP density, and TP ratio mixture distribution
        self.tp_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(phi_dim, n_bins)
        )
        self.fp_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(phi_dim, n_bins)
        )
        
        # Mixture of logistic components for TP ratio distribution
        self.num_mixture_components = 5  # Number of logistic components
        self.mixture_weights_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(phi_dim, self.num_mixture_components)
        )  # Mixture weights
        self.mixture_locations_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(phi_dim, self.num_mixture_components)
        )  # Location parameters μᵢ
        self.mixture_scales_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(phi_dim, self.num_mixture_components)
        )  # Scale parameters sᵢ
    
    def forward(self, x, counts):
        mask = torch.arange(x.size(1), device=x.device)[None, :] < counts[:, None]
        phi_out = self.phi(x * mask.unsqueeze(-1))
        query = self.query.repeat(x.shape[0], 1, 1)
        agg = self.pooling(query, phi_out).squeeze(1)
        rho_out = self.rho(agg)
        
        tp_out = F.softmax(self.tp_head(rho_out), dim=1)
        fp_out = F.softmax(self.fp_head(rho_out), dim=1)
        
        # Mixture of logistic components for TP ratio distribution
        mixture_weights_raw = self.mixture_weights_head(rho_out)  # (B, num_components)
        mixture_locations_raw = self.mixture_locations_head(rho_out)  # (B, num_components)
        mixture_scales_raw = self.mixture_scales_head(rho_out)  # (B, num_components)
        
        # Normalize mixture weights
        mixture_weights = F.softmax(mixture_weights_raw, dim=1)
        
        # Constrain locations to [0, 1] using sigmoid
        mixture_locations = torch.sigmoid(mixture_locations_raw)
        
        # Constrain scales to be positive using softplus
        mixture_scales = F.softplus(mixture_scales_raw) + 1e-6  # Add small constant for numerical stability
        
        return tp_out, fp_out, mixture_weights, mixture_locations, mixture_scales

def calculate_predicted_precision(tp_density, fp_density, mixture_weights, mixture_locations, mixture_scales, sample_size):
    """Calculate predicted TP ratio from mixture of logistic distributions"""
    # Calculate the mean of the mixture distribution
    # Mean of mixture = Σᵢ wᵢ * μᵢ (for logistic, mean ≈ location parameter)
    predicted_tp_ratio = torch.sum(mixture_weights * mixture_locations, dim=1)
    return predicted_tp_ratio

def mixture_logistic_cdf(x, weights, locations, scales):
    """
    Compute CDF of mixture of logistic distributions: P(X ≤ x) = Σᵢ wᵢ * σ((x - μᵢ) / sᵢ)
    """
    # x: (batch_size, 1) or (batch_size,)
    # weights: (batch_size, num_components)
    # locations: (batch_size, num_components) 
    # scales: (batch_size, num_components)
    
    if x.dim() == 1:
        x = x.unsqueeze(1)  # (batch_size, 1)
    
    # Compute (x - μᵢ) / sᵢ for all components
    standardized = (x - locations) / scales  # (batch_size, num_components)
    
    # Compute σ((x - μᵢ) / sᵢ) for all components
    logistic_cdfs = torch.sigmoid(standardized)  # (batch_size, num_components)
    
    # Weighted sum: Σᵢ wᵢ * σ((x - μᵢ) / sᵢ)
    mixture_cdf = torch.sum(weights * logistic_cdfs, dim=1)  # (batch_size,)
    
    return mixture_cdf

def mixture_logistic_pdf(x, weights, locations, scales):
    """
    Compute PDF of mixture of logistic distributions: p(x) = Σᵢ wᵢ * (1/sᵢ) * σ((x - μᵢ) / sᵢ) * (1 - σ((x - μᵢ) / sᵢ))
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)  # (batch_size, 1)
    
    # Compute (x - μᵢ) / sᵢ for all components
    standardized = (x - locations) / scales  # (batch_size, num_components)
    
    # Compute σ((x - μᵢ) / sᵢ) for all components
    logistic_cdfs = torch.sigmoid(standardized)  # (batch_size, num_components)
    
    # Logistic PDF = (1/s) * σ(z) * (1 - σ(z)) where z = (x - μ) / s
    logistic_pdfs = (1.0 / scales) * logistic_cdfs * (1.0 - logistic_cdfs)  # (batch_size, num_components)
    
    # Weighted sum: Σᵢ wᵢ * pdfᵢ(x)
    mixture_pdf = torch.sum(weights * logistic_pdfs, dim=1)  # (batch_size,)
    
    return mixture_pdf

def mixture_logistic_loss(mixture_weights, mixture_locations, mixture_scales, target_tp_ratio):
    """
    Negative log-likelihood loss for mixture of logistic distributions
    """
    # Clamp target to valid range [ε, 1-ε] for numerical stability
    target_clamped = torch.clamp(target_tp_ratio, 1e-6, 1-1e-6)
    
    # Compute PDF at target values
    pdf_values = mixture_logistic_pdf(target_clamped, mixture_weights, mixture_locations, mixture_scales)
    
    # Negative log-likelihood
    nll_loss = -torch.log(pdf_values + 1e-9).mean()
    
    return nll_loss

def precision_aware_loss(pred_tp, pred_fp, mixture_weights, mixture_locations, mixture_scales, target_tp, target_fp, target_ratio, target_precision, sample_sizes, 
                        density_weight=1.0, distribution_weight=2.0, precision_weight=1.0):
    """
    Combined loss with mixture of logistic distributions for TP ratio uncertainty
    """
    # Traditional density losses
    density_loss = F.mse_loss(pred_tp, target_tp) + F.mse_loss(pred_fp, target_fp)
    
    # Mixture logistic distribution loss (encourages proper uncertainty modeling)
    distribution_loss = mixture_logistic_loss(mixture_weights, mixture_locations, mixture_scales, target_precision)
    
    # Mean prediction loss (ensures mean is correct)
    predicted_precision = calculate_predicted_precision(pred_tp, pred_fp, mixture_weights, mixture_locations, mixture_scales, sample_sizes)
    precision_loss = F.mse_loss(predicted_precision, target_precision)
    
    # Combined loss with weights
    total_loss = (density_weight * density_loss + 
                  distribution_weight * distribution_loss + 
                  precision_weight * precision_loss)
    
    return total_loss, density_loss, distribution_loss, precision_loss

def collate_fn(batch):
    features, tps, fps, ratios, precisions, counts = zip(*[b for b in batch if b is not None])
    if not features: 
        return None, None, None, None, None, None
    
    max_len = max(len(f) for f in features)
    padded = torch.zeros(len(features), max_len, 3)
    for i, f in enumerate(features): 
        padded[i, :len(f), :] = f
    
    return (padded, torch.stack(tps), torch.stack(fps), 
            torch.stack(ratios), torch.stack(precisions), torch.tensor(counts, dtype=torch.float32))

def main(args):
    with open("ground_truth_histograms.pkl", 'rb') as f: 
        data = pickle.load(f)
    
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    train_ds = CameraDensityDatasetPrecisionAware(train_data, SAMPLE_SIZE_RANGE)
    val_ds = CameraDensityDatasetPrecisionAware(val_data, SAMPLE_SIZE_RANGE)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                             num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                           num_workers=4, collate_fn=collate_fn)
    
    model = DeepSetsPrecisionAware(args.phi_dim, N_BINS, args.num_heads, args.dropout_rate).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    plots_dir = "plots_precision_aware"
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Starting precision-aware training on {DEVICE}...")
    print(f"Hyperparameters: LR={args.learning_rate}, Dropout={args.dropout_rate}, Weight Decay={args.weight_decay}")
    
    # Initialize best score tracking
    best_precision_score = -1.0
    best_model_path = "runs/best_model/best_checkpoint_precision_aware.pth"
    os.makedirs("runs/best_model", exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0
        total_train_density_loss = 0
        total_train_distribution_loss = 0
        total_train_precision_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            if batch[0] is None: 
                continue
            
            features, gt_tp, gt_fp, gt_ratio, gt_precision, counts = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            pred_tp, pred_fp, mixture_weights, mixture_locations, mixture_scales = model(features, counts)
            
            total_loss, density_loss, distribution_loss, precision_loss = precision_aware_loss(
                pred_tp, pred_fp, mixture_weights, mixture_locations, mixture_scales, gt_tp, gt_fp, gt_ratio, gt_precision, counts,
                density_weight=args.density_weight, distribution_weight=args.distribution_weight, precision_weight=args.precision_weight
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += total_loss.item()
            total_train_density_loss += density_loss.item()
            total_train_distribution_loss += distribution_loss.item()
            total_train_precision_loss += precision_loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_density_loss = total_train_density_loss / len(train_loader)
        avg_train_distribution_loss = total_train_distribution_loss / len(train_loader)
        avg_train_precision_loss = total_train_precision_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        total_val_density_loss = 0
        total_val_distribution_loss = 0
        total_val_precision_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                if batch[0] is None: 
                    continue
                
                features, gt_tp, gt_fp, gt_ratio, gt_precision, counts = [b.to(DEVICE) for b in batch]
                pred_tp, pred_fp, mixture_weights, mixture_locations, mixture_scales = model(features, counts)
                
                total_loss, density_loss, distribution_loss, precision_loss = precision_aware_loss(
                    pred_tp, pred_fp, mixture_weights, mixture_locations, mixture_scales, gt_tp, gt_fp, gt_ratio, gt_precision, counts,
                    density_weight=args.density_weight, distribution_weight=args.distribution_weight, precision_weight=args.precision_weight
                )
                
                total_val_loss += total_loss.item()
                total_val_density_loss += density_loss.item()
                total_val_distribution_loss += distribution_loss.item()
                total_val_precision_loss += precision_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_density_loss = total_val_density_loss / len(val_loader)
        avg_val_distribution_loss = total_val_distribution_loss / len(val_loader)
        avg_val_precision_loss = total_val_precision_loss / len(val_loader)
        
        # Use precision loss as the main metric for model selection
        precision_score = -avg_val_precision_loss  # Negative because lower loss is better
        
        if precision_score > best_precision_score:
            best_precision_score = precision_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'precision_score': precision_score,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_precision_loss': avg_train_precision_loss,
                'val_precision_loss': avg_val_precision_loss,
                'hyperparameters': {
                    'learning_rate': args.learning_rate,
                    'batch_size': args.batch_size,
                    'phi_dim': args.phi_dim,
                    'num_heads': args.num_heads,
                    'epochs': args.epochs,
                    'density_weight': args.density_weight,
                    'distribution_weight': args.distribution_weight,
                    'precision_weight': args.precision_weight,
                    'dropout_rate': args.dropout_rate,
                    'weight_decay': args.weight_decay
                }
            }, best_model_path)
            print(f"Epoch {epoch} | Train: {avg_train_loss:.4f} (D:{avg_train_density_loss:.4f}, B:{avg_train_distribution_loss:.4f}, P:{avg_train_precision_loss:.4f}) | Val: {avg_val_loss:.4f} (D:{avg_val_density_loss:.4f}, B:{avg_val_distribution_loss:.4f}, P:{avg_val_precision_loss:.4f}) | ✅ BEST MODEL SAVED")
        else:
            print(f"Epoch {epoch} | Train: {avg_train_loss:.4f} (D:{avg_train_density_loss:.4f}, B:{avg_train_distribution_loss:.4f}, P:{avg_train_precision_loss:.4f}) | Val: {avg_val_loss:.4f} (D:{avg_val_density_loss:.4f}, B:{avg_val_distribution_loss:.4f}, P:{avg_val_precision_loss:.4f})")

    print(f"\n🎉 Precision-aware training complete!")
    print(f"Best model saved to: {best_model_path} (Precision Score = {best_precision_score:.4f})")

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