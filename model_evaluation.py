#!/usr/bin/env python3
"""
Model Evaluation Script for ML-Based Satisfaction Proxy Project

This script evaluates the trained DeepSets model on the test set,
calculates the R² score between predicted and ground truth curves,
and visualizes the results.
"""

import pandas as pd
import numpy as np
import torch
import pickle
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Import necessary classes from the training script
from model_training import DeepSetsModel, collate_variable_length, evaluate_camera
from tqdm import tqdm

def plot_evaluation_curves(all_gt_curves: Dict, all_pred_curves: Dict, output_dir: str):
    """
    Plot the aggregated ground truth vs predicted curves for TP and FP.
    
    Args:
        all_gt_curves: Aggregated ground truth curves
        all_pred_curves: Aggregated predicted curves
        output_dir: Directory to save the plot
    """
    
    # Prepare data for plotting
    plot_data = []
    
    for k, v in all_gt_curves['tp'].items():
        plot_data.append({'subset_size': k, 'count': v, 'type': 'TP', 'source': 'Ground Truth'})
    for k, v in all_gt_curves['fp'].items():
        plot_data.append({'subset_size': k, 'count': v, 'type': 'FP', 'source': 'Ground Truth'})
        
    for k, v in all_pred_curves['tp'].items():
        plot_data.append({'subset_size': k, 'count': v, 'type': 'TP', 'source': 'Predicted'})
    for k, v in all_pred_curves['fp'].items():
        plot_data.append({'subset_size': k, 'count': v, 'type': 'FP', 'source': 'Predicted'})
        
    plot_df = pd.DataFrame(plot_data)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # TP Plot
    sns.lineplot(data=plot_df[plot_df['type'] == 'TP'], x='subset_size', y='count', hue='source', style='source', markers=True, ax=axes[0])
    axes[0].set_title('True Positive (TP) Curves', fontsize=16)
    axes[0].set_xlabel('Subset Size (k)', fontsize=12)
    axes[0].set_ylabel('Average Count', fontsize=12)
    axes[0].legend(title='Source')
    
    # FP Plot
    sns.lineplot(data=plot_df[plot_df['type'] == 'FP'], x='subset_size', y='count', hue='source', style='source', markers=True, ax=axes[1])
    axes[1].set_title('False Positive (FP) Curves', fontsize=16)
    axes[1].set_xlabel('Subset Size (k)', fontsize=12)
    axes[1].set_ylabel('') # Remove y-label for clarity
    axes[1].legend(title='Source')
    
    fig.suptitle('Model Evaluation: Predicted vs. Ground Truth Curves', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(Path(output_dir) / 'evaluation_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📁 Evaluation curves plot saved to {output_dir}/evaluation_curves.png")
    
def main():
    """Main function to run the model evaluation."""
    
    # --- Configuration ---
    MODEL_PATH = 'final_model.pth'
    DATA_PATH = 'processed_theft_data.parquet'
    SPLITS_PATH = 'data_splits.pkl'
    OUTPUT_DIR = 'evaluation_results'
    EVAL_SUBSET_SIZES = [10, 20, 50, 100, 150, 200] # Subset sizes for evaluation
    N_SAMPLES_PER_SIZE = 100 # Samples per subset size for stable estimates
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Check for necessary files
    if not all(Path(p).exists() for p in [MODEL_PATH, DATA_PATH, SPLITS_PATH]):
        print("Error: Model, data, or splits file not found. Ensure training is complete.")
        return
        
    # --- Load Model and Data ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = DeepSetsModel(phi_dim=64, rho_dims=[128, 64], n_components=3, device=device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    print("✅ Model loaded successfully.")
    
    # Load data
    df = pd.read_parquet(DATA_PATH)
    df['camera_id_combined'] = df['camera_id'].astype(str) + '_' + df['store_id'].astype(str)
    
    with open(SPLITS_PATH, 'rb') as f:
        data_splits = pickle.load(f)
    
    test_files = data_splits['test_files']
    
    # We don't load all test data into memory at once.
    # The evaluation function will load one camera at a time.
    print(f"✅ Ready to evaluate on {len(test_files)} test camera files.")
    
    # --- Run Evaluation ---
    all_gt_tp = {}
    all_gt_fp = {}
    all_pred_tp = {}
    all_pred_fp = {}
    
    print(f"\nRunning evaluation on {len(test_files)} test cameras...")
    
    for i, file_path in enumerate(tqdm(test_files, desc="Evaluating test cameras")):
        try:
            camera_df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"Warning: Could not read {file_path}. Skipping. Error: {e}")
            continue

        eval_subset_sizes = np.linspace(10, min(len(camera_df), 500), num=10, dtype=int)
        if len(eval_subset_sizes) == 0 or eval_subset_sizes[0] == 0: continue
        
        gt_curves, pred_curves = evaluate_camera(model, camera_df, eval_subset_sizes, N_SAMPLES_PER_SIZE, device)
        
        # Aggregate results
        for k, v in gt_curves['tp'].items():
            all_gt_tp.setdefault(k, []).append(v)
        for k, v in gt_curves['fp'].items():
            all_gt_fp.setdefault(k, []).append(v)
            
        for k, v in pred_curves['tp'].items():
            all_pred_tp.setdefault(k, []).append(v)
        for k, v in pred_curves['fp'].items():
            all_pred_fp.setdefault(k, []).append(v)
            
    # Average the curves across all cameras
    avg_gt_tp = {k: np.mean(v) for k, v in all_gt_tp.items()}
    avg_gt_fp = {k: np.mean(v) for k, v in all_gt_fp.items()}
    avg_pred_tp = {k: np.mean(v) for k, v in all_pred_tp.items()}
    avg_pred_fp = {k: np.mean(v) for k, v in all_pred_fp.items()}
    
    all_gt_curves = {'tp': avg_gt_tp, 'fp': avg_gt_fp}
    all_pred_curves = {'tp': avg_pred_tp, 'fp': avg_pred_fp}
    
    # --- Calculate R² Score ---
    gt_tp_values = np.array(list(avg_gt_tp.values()))
    pred_tp_values = np.array(list(avg_pred_tp.values()))
    gt_fp_values = np.array(list(avg_gt_fp.values()))
    pred_fp_values = np.array(list(avg_pred_fp.values()))
    
    r2_tp = r2_score(gt_tp_values, pred_tp_values)
    r2_fp = r2_score(gt_fp_values, pred_fp_values)
    
    print("\n--- Evaluation Results ---")
    print(f"R² Score (TP): {r2_tp:.4f}")
    print(f"R² Score (FP): {r2_fp:.4f}")
    print("--------------------------\n")
    
    # --- Visualize Results ---
    plot_evaluation_curves(all_gt_curves, all_pred_curves, OUTPUT_DIR)
    
    print("✅ Evaluation complete.")

if __name__ == "__main__":
    main() 