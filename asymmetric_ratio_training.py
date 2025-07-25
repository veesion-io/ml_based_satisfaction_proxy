#!/usr/bin/env python3
"""
Asymmetric Ratio Loss Training
Implements asymmetric loss on TP ratio to bias predictions for better precision accuracy
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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import the dataset and model from the ratio training script
from density_prediction_training_with_ratio import (
    CameraDensityDatasetWithRatio, 
    ResidualMLP, 
    MAB,
    collate_fn,
    plot_densities_with_ratio
)

class DeepSetsAsymmetricRatio(nn.Module):
    def __init__(self, phi_dim, n_bins, num_heads):
        super(DeepSetsAsymmetricRatio, self).__init__()
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
        
        self.tp_head = nn.Linear(phi_dim, n_bins)
        self.fp_head = nn.Linear(phi_dim, n_bins)
        self.ratio_head = nn.Linear(phi_dim, 1)
    
    def forward(self, x, counts):
        mask = torch.arange(x.size(1), device=x.device)[None, :] < counts[:, None]
        phi_out = self.phi(x * mask.unsqueeze(-1))
        query = self.query.repeat(x.shape[0], 1, 1)
        agg = self.pooling(query, phi_out).squeeze(1)
        rho_out = self.rho(agg)
        
        tp_out = F.softmax(self.tp_head(rho_out), dim=1)
        fp_out = F.softmax(self.fp_head(rho_out), dim=1)
        ratio_out = torch.sigmoid(self.ratio_head(rho_out)).squeeze(-1)
        
        return tp_out, fp_out, ratio_out

def asymmetric_ratio_loss(pred_ratio, target_ratio, over_penalty=2.0, under_penalty=1.0):
    """
    Asymmetric loss that penalizes over-prediction more than under-prediction
    This should bias the model toward slightly under-predicting ratios
    
    Args:
        pred_ratio: Predicted TP ratios
        target_ratio: Ground truth TP ratios  
        over_penalty: Penalty multiplier for over-prediction (pred > target)
        under_penalty: Penalty multiplier for under-prediction (pred < target)
    """
    errors = pred_ratio - target_ratio
    
    # Split into over-predictions and under-predictions
    over_mask = errors > 0
    under_mask = errors <= 0
    
    # Apply different penalties
    over_loss = over_penalty * (errors[over_mask] ** 2)
    under_loss = under_penalty * (errors[under_mask] ** 2)
    
    # Combine losses
    total_loss = torch.cat([over_loss, under_loss]).mean() if len(over_loss) > 0 and len(under_loss) > 0 else \
                 over_loss.mean() if len(over_loss) > 0 else \
                 under_loss.mean() if len(under_loss) > 0 else \
                 torch.tensor(0.0, device=pred_ratio.device)
    
    return total_loss

def combined_asymmetric_loss(pred_tp, pred_fp, pred_ratio, target_tp, target_fp, target_ratio, 
                           density_weight=1.0, ratio_weight=1.0, over_penalty=3.0, under_penalty=1.0):
    """
    Combined loss with asymmetric ratio penalty
    """
    # Standard density loss
    density_loss = F.mse_loss(pred_tp, target_tp) + F.mse_loss(pred_fp, target_fp)
    
    # Asymmetric ratio loss
    ratio_loss = asymmetric_ratio_loss(pred_ratio, target_ratio, over_penalty, under_penalty)
    
    # Combined loss
    total_loss = density_weight * density_loss + ratio_weight * ratio_loss
    
    return total_loss, density_loss, ratio_loss

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
    
    model = DeepSetsAsymmetricRatio(args.phi_dim, N_BINS, args.num_heads).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    plots_dir = "plots_asymmetric"
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Starting asymmetric ratio training on {DEVICE}...")
    print(f"Over-penalty: {args.over_penalty}x, Under-penalty: {args.under_penalty}x")
    
    # Initialize best score tracking
    best_r2_score = -1.0
    best_model_path = "runs/best_model/best_checkpoint_asymmetric.pth"
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
            
            total_loss, density_loss, ratio_loss = combined_asymmetric_loss(
                pred_tp, pred_fp, pred_ratio, gt_tp, gt_fp, gt_ratio,
                density_weight=args.density_weight, 
                ratio_weight=args.ratio_weight,
                over_penalty=args.over_penalty,
                under_penalty=args.under_penalty
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += total_loss.item()
            total_train_density_loss += density_loss.item()
            total_train_ratio_loss += ratio_loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_density_loss = total_train_density_loss / len(train_loader)
        avg_train_ratio_loss = total_train_ratio_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        total_val_density_loss = 0
        total_val_ratio_loss = 0
        
        # For R² calculation
        all_pred_tp, all_pred_fp, all_pred_ratio = [], [], []
        all_gt_tp, all_gt_fp, all_gt_ratio = [], [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                if batch[0] is None: 
                    continue
                
                features, gt_tp, gt_fp, gt_ratio, counts = [b.to(DEVICE) for b in batch]
                pred_tp, pred_fp, pred_ratio = model(features, counts)
                
                total_loss, density_loss, ratio_loss = combined_asymmetric_loss(
                    pred_tp, pred_fp, pred_ratio, gt_tp, gt_fp, gt_ratio,
                    density_weight=args.density_weight, 
                    ratio_weight=args.ratio_weight,
                    over_penalty=args.over_penalty,
                    under_penalty=args.under_penalty
                )
                
                total_val_loss += total_loss.item()
                total_val_density_loss += density_loss.item()
                total_val_ratio_loss += ratio_loss.item()
                
                # Collect for R² calculation
                all_pred_tp.append(pred_tp.cpu())
                all_pred_fp.append(pred_fp.cpu())
                all_pred_ratio.append(pred_ratio.cpu())
                all_gt_tp.append(gt_tp.cpu())
                all_gt_fp.append(gt_fp.cpu())
                all_gt_ratio.append(gt_ratio.cpu())
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_density_loss = total_val_density_loss / len(val_loader)
        avg_val_ratio_loss = total_val_ratio_loss / len(val_loader)
        
        # Calculate R² scores
        try:
            from sklearn.metrics import r2_score
            
            pred_tp_flat = torch.cat(all_pred_tp, dim=0).numpy().flatten()
            pred_fp_flat = torch.cat(all_pred_fp, dim=0).numpy().flatten()
            pred_ratio_flat = torch.cat(all_pred_ratio, dim=0).numpy().flatten()
            
            gt_tp_flat = torch.cat(all_gt_tp, dim=0).numpy().flatten()
            gt_fp_flat = torch.cat(all_gt_fp, dim=0).numpy().flatten()
            gt_ratio_flat = torch.cat(all_gt_ratio, dim=0).numpy().flatten()
            
            r2_tp = r2_score(gt_tp_flat, pred_tp_flat)
            r2_fp = r2_score(gt_fp_flat, pred_fp_flat)
            r2_ratio = r2_score(gt_ratio_flat, pred_ratio_flat)
            r2_overall = (r2_tp + r2_fp) / 2.0  # Average R² for density prediction
            
        except Exception as e:
            r2_tp = r2_fp = r2_ratio = r2_overall = 0.0
        
        # Save best model based on overall R²
        if r2_overall > best_r2_score:
            best_r2_score = r2_overall
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'r2_score': r2_overall,
                'r2_tp': r2_tp,
                'r2_fp': r2_fp,
                'r2_ratio': r2_ratio,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'hyperparameters': {
                    'learning_rate': args.learning_rate,
                    'batch_size': args.batch_size,
                    'phi_dim': args.phi_dim,
                    'num_heads': args.num_heads,
                    'epochs': args.epochs,
                    'density_weight': args.density_weight,
                    'ratio_weight': args.ratio_weight,
                    'over_penalty': args.over_penalty,
                    'under_penalty': args.under_penalty
                }
            }, best_model_path)
            print(f"Epoch {epoch} | Train: {avg_train_loss:.4f} (D:{avg_train_density_loss:.4f}, R:{avg_train_ratio_loss:.4f}) | Val: {avg_val_loss:.4f} (D:{avg_val_density_loss:.4f}, R:{avg_val_ratio_loss:.4f}) | R²: {r2_overall:.4f} (TP:{r2_tp:.4f}, FP:{r2_fp:.4f}, Ratio:{r2_ratio:.4f}) | ✅ BEST")
        else:
            print(f"Epoch {epoch} | Train: {avg_train_loss:.4f} (D:{avg_train_density_loss:.4f}, R:{avg_train_ratio_loss:.4f}) | Val: {avg_val_loss:.4f} (D:{avg_val_density_loss:.4f}, R:{avg_val_ratio_loss:.4f}) | R²: {r2_overall:.4f} (TP:{r2_tp:.4f}, FP:{r2_fp:.4f}, Ratio:{r2_ratio:.4f})")

    print(f"\n🎉 Asymmetric ratio training complete!")
    print(f"Best model saved to: {best_model_path} (R² = {best_r2_score:.4f})")
    print(f"Final R2 Score: {best_r2_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--phi_dim', type=int, default=DEFAULT_PHI_DIM)
    parser.add_argument('--num_heads', type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument('--density_weight', type=float, default=1.0, help='Weight for density loss')
    parser.add_argument('--ratio_weight', type=float, default=1.0, help='Weight for ratio loss')
    parser.add_argument('--over_penalty', type=float, default=3.0, help='Penalty for over-predicting ratio')
    parser.add_argument('--under_penalty', type=float, default=1.0, help='Penalty for under-predicting ratio')
    main(parser.parse_args()) 