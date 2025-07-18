#!/usr/bin/env python3
"""
Density Prediction Model Training Pipeline

This script implements a DeepSets model to predict the probability density
of True Positives (TP) and False Positives (FP) for a given camera, based on
a small sample of its alert probabilities.
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
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

# --- Configuration ---
N_BINS = 100 # Must match the preprocessing script
SAMPLE_SIZE_RANGE = (10, 50) # Use a range for input sample sizes
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CameraDensityDataset(Dataset):
    """
    Dataset that loads pre-calculated ground truth densities and provides
    random samples of alert probabilities for model input.
    """
    def __init__(self, processed_data: List[Dict], sample_size_range: Tuple[int, int]):
        self.sample_size_range = sample_size_range
        print("Loading pre-calculated densities...")
        # Load the data directly, no on-the-fly calculation
        self.camera_data = processed_data
        print(f"Loaded {len(self.camera_data)} cameras with valid densities.")

    def __len__(self):
        return len(self.camera_data)

    def __getitem__(self, idx):
        camera = self.camera_data[idx]
        
        # We still need to read the parquet to get the probability samples
        df = pd.read_parquet(camera['file_path'])

        # Dynamically choose a random sample size k
        min_size, max_size = self.sample_size_range
        # Ensure k is not larger than the number of alerts available
        k = random.randint(min_size, min(max_size, len(df)))

        sample_probs = df['max_proba'].sample(n=k, replace=True).values
        sample_probs = torch.tensor(sample_probs, dtype=torch.float32).unsqueeze(-1)
        
        tp_density = torch.tensor(camera['tp_density'], dtype=torch.float32)
        fp_density = torch.tensor(camera['fp_density'], dtype=torch.float32)
        
        return sample_probs, tp_density, fp_density

class DeepSetsDensity(nn.Module):
    """DeepSets model to predict TP/FP densities."""
    def __init__(self, phi_dim: int = 128, rho_dims: List[int] = [256, 128], n_bins: int = N_BINS):
        super(DeepSetsDensity, self).__init__()
        
        self.phi = nn.Sequential(
            nn.Linear(1, phi_dim), nn.ReLU(),
            nn.Linear(phi_dim, phi_dim), nn.ReLU()
        )
        
        rho_layers = []
        input_dim = phi_dim
        for dim in rho_dims:
            rho_layers.extend([nn.Linear(input_dim, dim), nn.ReLU()])
            input_dim = dim
            
        self.rho = nn.Sequential(*rho_layers)
        
        # Two heads: one for TP density, one for FP density
        self.tp_head = nn.Linear(input_dim, n_bins)
        self.fp_head = nn.Linear(input_dim, n_bins)

    def forward(self, x):
        phi_out = self.phi(x)
        aggregated = torch.mean(phi_out, dim=1)
        rho_out = self.rho(aggregated)
        
        tp_logits = self.tp_head(rho_out)
        fp_logits = self.fp_head(rho_out)
        
        return tp_logits, fp_logits

def kl_loss(predicted_logits, target_density):
    """KL Divergence loss for comparing predicted logits to a target density."""
    # Use log_softmax for numerical stability
    log_pred_density = F.log_softmax(predicted_logits, dim=-1)
    # Target density is already a distribution, so no log is needed
    return F.kl_div(log_pred_density, target_density, reduction='batchmean')

def collate_variable_size(batch):
    """Pads variable-sized probability samples to the max length in a batch."""
    # This function is now necessary because our inputs have random sizes.
    # Filter out None items that might come from a dataset __getitem__ method
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None

    probs_list, tp_list, fp_list = zip(*batch)
    
    max_len = max(len(p) for p in probs_list)
    
    # Pad the probability tensors
    padded_probs = torch.zeros(len(probs_list), max_len, 1)
    for i, p in enumerate(probs_list):
        padded_probs[i, :len(p), :] = p
        
    return padded_probs, torch.stack(tp_list), torch.stack(fp_list)


def plot_density_comparison(epoch, plot_data_list, n_bins, output_dir):
    """
    Plots and saves a 5x2 grid comparison of ground truth, predicted, and subset densities.
    """
    fig, axes = plt.subplots(5, 2, figsize=(18, 22), squeeze=False)
    fig.suptitle(f'Epoch {epoch}: Multi-Camera Evaluation', fontsize=20)
    
    x_axis = np.linspace(0, 1, n_bins)
    width = x_axis[1] - x_axis[0]

    for i, data in enumerate(plot_data_list):
        k = data['k']
        gt_tp = data['gt_tp']
        pred_tp = data['pred_tp']
        gt_fp = data['gt_fp']
        pred_fp = data['pred_fp']
        subset_hist = data['subset_hist']

        # --- TP Density Plot ---
        ax_tp = axes[i, 0]
        ax_tp.bar(x_axis, subset_hist, width=width, label='Subset Histogram', color='green', alpha=0.5)
        ax_tp.plot(x_axis, gt_tp, label='Ground Truth (KDE)', color='blue', linewidth=2)
        ax_tp.plot(x_axis, pred_tp, label='Predicted Density', color='red', linestyle='--', linewidth=2)
        ax_tp.set_title(f'Camera {i+1} TP Density (Sample Size k={k})')
        ax_tp.set_xlabel('Probability')
        ax_tp.set_ylabel('Density')
        ax_tp.legend()
        ax_tp.grid(True, linestyle='--', alpha=0.6)

        # --- FP Density Plot ---
        ax_fp = axes[i, 1]
        ax_fp.bar(x_axis, subset_hist, width=width, label='Subset Histogram', color='green', alpha=0.5)
        ax_fp.plot(x_axis, gt_fp, label='Ground Truth (KDE)', color='blue', linewidth=2)
        ax_fp.plot(x_axis, pred_fp, label='Predicted Density', color='red', linestyle='--', linewidth=2)
        ax_fp.set_title(f'Camera {i+1} FP Density (Sample Size k={k})')
        ax_fp.set_xlabel('Probability')
        ax_fp.legend()
        ax_fp.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plot_path = os.path.join(output_dir, f"density_eval_epoch_{epoch}.png")
    plt.savefig(plot_path)
    plt.close(fig)
    return plot_path

def main():
    """Main training and evaluation pipeline."""
    # Load the pre-processed ground truth data
    processed_data_file = "ground_truth_densities.pkl"
    if not os.path.exists(processed_data_file):
        print(f"Error: Processed data file not found at {processed_data_file}")
        print("Please run 'preprocess_densities.py' first.")
        return

    with open(processed_data_file, 'rb') as f:
        all_processed_data = pickle.load(f)

    train_data, val_data = train_test_split(all_processed_data, test_size=0.2, random_state=42)

    train_dataset = CameraDensityDataset(train_data, SAMPLE_SIZE_RANGE)
    val_dataset = CameraDensityDataset(val_data, SAMPLE_SIZE_RANGE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_variable_size)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_variable_size)
    
    model = DeepSetsDensity().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter('runs/density_prediction')
    plots_dir = "density_evaluation_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Starting training on {DEVICE}...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            probs, gt_tp, gt_fp = [b.to(DEVICE) for b in batch]
            
            optimizer.zero_grad()
            pred_tp_logits, pred_fp_logits = model(probs)
            
            loss_tp = kl_loss(pred_tp_logits, gt_tp)
            loss_fp = kl_loss(pred_fp_logits, gt_fp)
            loss = loss_tp + loss_fp
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # --- Validation and Plotting ---
        model.eval()
        with torch.no_grad():
            # --- For Loss Calculation ---
            # Get a single batch to calculate an average validation loss
            batch = next(iter(val_loader), None)
            if batch is None:
                avg_val_loss = 0
            else:
                val_probs_batch, val_gt_tp_batch, val_gt_fp_batch = [b.to(DEVICE) for b in batch]
                pred_tp_logits_batch, pred_fp_logits_batch = model(val_probs_batch)
                val_loss_tp = kl_loss(pred_tp_logits_batch, val_gt_tp_batch)
                val_loss_fp = kl_loss(pred_fp_logits_batch, val_gt_fp_batch)
                avg_val_loss = (val_loss_tp + val_loss_fp).item()
            
            # --- For Plotting ---
            # Generate 5 random examples for the composite plot
            plot_data_list = []
            num_plot_samples = min(5, len(val_dataset))
            random_cameras_for_plot = random.sample(val_dataset.camera_data, num_plot_samples)

            for cam_data in random_cameras_for_plot:
                val_df = pd.read_parquet(cam_data['file_path'])

                min_size, max_size = val_dataset.sample_size_range
                k = random.randint(min_size, min(max_size, len(val_df)))
                
                # Get the small sample of probabilities for model input
                val_probs_np = val_df['max_proba'].sample(n=k, replace=True).values
                
                # Create the "naive" histogram from this small sample
                subset_hist, _ = np.histogram(val_probs_np, bins=N_BINS, range=(0, 1))
                subset_hist_density = subset_hist / (subset_hist.sum() + 1e-6)
                
                # Get the model's prediction from the small sample
                val_probs_tensor = torch.tensor(val_probs_np, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(DEVICE)
                pred_tp_logits, pred_fp_logits = model(val_probs_tensor)
                
                pred_tp_density = F.softmax(pred_tp_logits.squeeze(0), dim=-1).cpu().numpy()
                pred_fp_density = F.softmax(pred_fp_logits.squeeze(0), dim=-1).cpu().numpy()

                plot_data_list.append({
                    'k': k,
                    'gt_tp': cam_data['tp_density'],
                    'pred_tp': pred_tp_density,
                    'gt_fp': cam_data['fp_density'],
                    'pred_fp': pred_fp_density,
                    'subset_hist': subset_hist_density
                })
            
            # Generate and save the composite plot
            if plot_data_list:
                plot_path = plot_density_comparison(epoch, plot_data_list, N_BINS, plots_dir)
            else:
                plot_path = "No plot generated."

        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        if os.path.exists(plot_path):
            writer.add_image('Evaluation/density_plot_grid', plt.imread(plot_path).transpose(2, 0, 1), epoch)
        
        print(f"Epoch {epoch}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Plot saved to {plot_path}")

    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    main() 