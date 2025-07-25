#!/usr/bin/env python3
"""
Model Convergence Evaluation Script

This script evaluates the model's performance on random camera subsets
with varying sample sizes (1% to 100% of camera data) and plots the
convergence of mean precision per camera.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import random
from typing import List, Tuple, Dict
from sklearn.metrics import r2_score
import argparse

from load_best_model import load_best_model, predict_densities
from density_prediction_training import N_BINS

def evaluate_camera_at_percentage(model, camera_data: pd.DataFrame, percentage: float, n_trials: int = 10) -> Dict:
    """
    Evaluate model performance on a camera at a specific percentage of data
    
    Args:
        model: Trained model
        camera_data: Camera DataFrame with 'max_proba' and 'is_theft' columns
        percentage: Percentage of data to use (0.01 to 1.0)
        n_trials: Number of random trials to average over
    
    Returns:
        Dictionary with performance metrics
    """
    sample_size = max(1, int(len(camera_data) * percentage))
    
    # Multiple trials for stable estimates
    r2_scores = []
    tp_r2_scores = []
    fp_r2_scores = []
    
    for trial in range(n_trials):
        try:
            # Sample data
            if sample_size >= len(camera_data):
                sample_data = camera_data
            else:
                sample_data = camera_data.sample(n=sample_size, replace=False)
            
            # Predict densities
            pred_tp, pred_fp = predict_densities(model, sample_data, sample_size=len(sample_data))
            
            # Calculate ground truth densities for this camera
            # (This is a simplified version - in practice you'd use the preprocessed ground truth)
            tp_probs = camera_data[camera_data['is_theft'] == 1]['max_proba'].values
            fp_probs = camera_data[camera_data['is_theft'] == 0]['max_proba'].values
            
            if len(tp_probs) < 5 or len(fp_probs) < 5:
                continue  # Skip cameras with insufficient data
            
            # Create ground truth histograms
            tp_hist, _ = np.histogram(tp_probs, bins=N_BINS, range=(0, 1))
            fp_hist, _ = np.histogram(fp_probs, bins=N_BINS, range=(0, 1))
            
            # Normalize to densities
            gt_tp = tp_hist / (tp_hist.sum() + 1e-9)
            gt_fp = fp_hist / (fp_hist.sum() + 1e-9)
            
            # Calculate R² scores
            if np.sum(gt_tp) > 0 and np.sum(pred_tp) > 0:
                tp_r2 = r2_score(gt_tp, pred_tp)
                tp_r2_scores.append(max(0, tp_r2))  # Clip negative R² to 0
            
            if np.sum(gt_fp) > 0 and np.sum(pred_fp) > 0:
                fp_r2 = r2_score(gt_fp, pred_fp)
                fp_r2_scores.append(max(0, fp_r2))  # Clip negative R² to 0
            
            if len(tp_r2_scores) > 0 and len(fp_r2_scores) > 0:
                combined_r2 = (tp_r2_scores[-1] + fp_r2_scores[-1]) / 2
                r2_scores.append(combined_r2)
        
        except Exception as e:
            # Skip failed trials
            continue
    
    # Return average metrics
    return {
        'percentage': percentage,
        'sample_size': sample_size,
        'mean_r2': np.mean(r2_scores) if r2_scores else 0.0,
        'std_r2': np.std(r2_scores) if r2_scores else 0.0,
        'mean_tp_r2': np.mean(tp_r2_scores) if tp_r2_scores else 0.0,
        'mean_fp_r2': np.mean(fp_r2_scores) if fp_r2_scores else 0.0,
        'n_successful_trials': len(r2_scores)
    }

def evaluate_convergence(model, camera_files: List[Path], percentages: np.ndarray, 
                        n_cameras: int = 50, n_trials_per_camera: int = 5) -> pd.DataFrame:
    """
    Evaluate model convergence across different sample sizes
    
    Args:
        model: Trained model
        camera_files: List of camera file paths
        percentages: Array of percentages to test (0.01 to 1.0)
        n_cameras: Number of random cameras to evaluate
        n_trials_per_camera: Number of trials per camera per percentage
    
    Returns:
        DataFrame with convergence results
    """
    # Select random cameras
    selected_cameras = random.sample(camera_files, min(n_cameras, len(camera_files)))
    
    results = []
    
    print(f"Evaluating {len(selected_cameras)} cameras across {len(percentages)} percentages...")
    
    for camera_file in tqdm(selected_cameras, desc="Processing cameras"):
        try:
            # Load camera data
            camera_data = pd.read_parquet(camera_file)
            
            # Skip cameras with insufficient data
            if len(camera_data) < 50:
                continue
                
            tp_count = camera_data['is_theft'].sum()
            fp_count = len(camera_data) - tp_count
            
            if tp_count < 5 or fp_count < 5:
                continue
            
            camera_name = camera_file.stem
            
            # Evaluate at each percentage
            for percentage in percentages:
                metrics = evaluate_camera_at_percentage(
                    model, camera_data, percentage, n_trials_per_camera
                )
                
                metrics['camera'] = camera_name
                metrics['camera_size'] = len(camera_data)
                results.append(metrics)
        
        except Exception as e:
            print(f"Error processing {camera_file}: {e}")
            continue
    
    return pd.DataFrame(results)

def plot_convergence(results_df: pd.DataFrame, output_path: str = "convergence_analysis.png"):
    """
    Create convergence plot showing mean precision per camera
    
    Args:
        results_df: DataFrame with convergence results
        output_path: Path to save the plot
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Calculate mean and std across cameras for each percentage
    convergence_stats = results_df.groupby('percentage').agg({
        'mean_r2': ['mean', 'std', 'count'],
        'mean_tp_r2': ['mean', 'std'],
        'mean_fp_r2': ['mean', 'std'],
        'sample_size': 'mean'
    }).reset_index()
    
    # Flatten column names
    convergence_stats.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                for col in convergence_stats.columns.values]
    
    percentages = convergence_stats['percentage'] * 100  # Convert to percentage
    
    # Main convergence plot
    ax1 = axes[0, 0]
    mean_r2 = convergence_stats['mean_r2_mean']
    std_r2 = convergence_stats['mean_r2_std']
    
    ax1.plot(percentages, mean_r2, 'b-', linewidth=2, label='Mean R²')
    ax1.fill_between(percentages, mean_r2 - std_r2, mean_r2 + std_r2, 
                     alpha=0.3, color='blue', label='±1 Std Dev')
    
    ax1.set_xlabel('Sample Size (% of Camera Data)')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Model Performance Convergence\n(Mean ± Std across cameras)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1)
    
    # TP vs FP convergence
    ax2 = axes[0, 1]
    ax2.plot(percentages, convergence_stats['mean_tp_r2_mean'], 'r-', 
             linewidth=2, label='True Positive R²')
    ax2.plot(percentages, convergence_stats['mean_fp_r2_mean'], 'g-', 
             linewidth=2, label='False Positive R²')
    
    ax2.fill_between(percentages, 
                     convergence_stats['mean_tp_r2_mean'] - convergence_stats['mean_tp_r2_std'],
                     convergence_stats['mean_tp_r2_mean'] + convergence_stats['mean_tp_r2_std'],
                     alpha=0.2, color='red')
    ax2.fill_between(percentages,
                     convergence_stats['mean_fp_r2_mean'] - convergence_stats['mean_fp_r2_std'],
                     convergence_stats['mean_fp_r2_mean'] + convergence_stats['mean_fp_r2_std'],
                     alpha=0.2, color='green')
    
    ax2.set_xlabel('Sample Size (% of Camera Data)')
    ax2.set_ylabel('R² Score')
    ax2.set_title('TP vs FP Performance Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1)
    
    # Sample size vs performance scatter
    ax3 = axes[1, 0]
    scatter_data = results_df[results_df['percentage'].isin([0.05, 0.1, 0.25, 0.5, 1.0])]
    
    for pct in [0.05, 0.1, 0.25, 0.5, 1.0]:
        subset = scatter_data[scatter_data['percentage'] == pct]
        if len(subset) > 0:
            ax3.scatter(subset['sample_size'], subset['mean_r2'], 
                       alpha=0.6, label=f'{int(pct*100)}%', s=30)
    
    ax3.set_xlabel('Absolute Sample Size')
    ax3.set_ylabel('R² Score')
    ax3.set_title('Sample Size vs Performance')
    ax3.legend(title='% of Camera Data')
    ax3.grid(True, alpha=0.3)
    
    # Convergence rate analysis
    ax4 = axes[1, 1]
    
    # Calculate convergence rate (change in R² per percentage point)
    if len(convergence_stats) > 1:
        conv_rate = np.diff(convergence_stats['mean_r2_mean']) / np.diff(convergence_stats['percentage'])
        actual_percentages = convergence_stats['percentage'] * 100
        pct_midpoints = (actual_percentages.iloc[1:].values + actual_percentages.iloc[:-1].values) / 2
        
        ax4.plot(pct_midpoints, conv_rate, 'purple', linewidth=2, marker='o')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Sample Size (% of Camera Data)')
    ax4.set_ylabel('Convergence Rate (ΔR² / Δ%)')
    ax4.set_title('Rate of Performance Improvement')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show for headless operation
    
    print(f"Convergence plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate model convergence across sample sizes')
    parser.add_argument('--n_cameras', type=int, default=50, 
                       help='Number of random cameras to evaluate')
    parser.add_argument('--n_trials', type=int, default=5,
                       help='Number of trials per camera per percentage')
    parser.add_argument('--output', type=str, default='convergence_analysis.png',
                       help='Output plot filename')
    parser.add_argument('--save_results', type=str, default='convergence_results.csv',
                       help='Save detailed results to CSV')
    
    args = parser.parse_args()
    
    # Load the best model
    print("Loading best model...")
    model, info = load_best_model()
    
    # Get camera files
    camera_files = list(Path("data_by_camera").glob("*.parquet"))
    print(f"Found {len(camera_files)} camera files")
    
    if len(camera_files) == 0:
        print("Error: No camera files found in data_by_camera/")
        return
    
    # Define percentage range (1% to 100%)
    percentages = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 
                           0.6, 0.7, 0.8, 0.9, 1.0])
    
    # Evaluate convergence
    results_df = evaluate_convergence(
        model, camera_files, percentages, 
        n_cameras=args.n_cameras, 
        n_trials_per_camera=args.n_trials
    )
    
    if len(results_df) == 0:
        print("Error: No evaluation results generated")
        return
    
    print(f"\nEvaluation complete! Processed {results_df['camera'].nunique()} cameras")
    print(f"Generated {len(results_df)} data points")
    
    # Save detailed results
    results_df.to_csv(args.save_results, index=False)
    print(f"Detailed results saved to: {args.save_results}")
    
    # Create convergence plot
    plot_convergence(results_df, args.output)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS SUMMARY")
    print("="*60)
    
    summary = results_df.groupby('percentage').agg({
        'mean_r2': ['mean', 'std'],
        'sample_size': 'mean'
    }).round(4)
    
    print("\nPerformance by Sample Size:")
    print(summary)
    
    # Find optimal sample size (where improvement rate drops below threshold)
    convergence_stats = results_df.groupby('percentage')['mean_r2'].mean()
    optimal_pct = None
    
    for i in range(1, len(convergence_stats)):
        pct_improvement = (convergence_stats.iloc[i] - convergence_stats.iloc[i-1]) / convergence_stats.iloc[i-1]
        if pct_improvement < 0.02:  # Less than 2% improvement
            optimal_pct = convergence_stats.index[i-1]
            break
    
    if optimal_pct:
        print(f"\n🎯 Optimal sample size: {optimal_pct*100:.0f}% of camera data")
        print(f"   (Performance plateaus with minimal improvement beyond this point)")

if __name__ == "__main__":
    main() 