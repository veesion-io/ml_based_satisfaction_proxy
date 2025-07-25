#!/usr/bin/env python3
"""
Improved Model Convergence Evaluation Script

This script evaluates the model's performance using the preprocessed KDE ground truth
densities for more accurate convergence analysis.
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
import pickle
from typing import List, Tuple, Dict
from sklearn.metrics import r2_score
import argparse

from load_best_model import load_best_model, predict_densities
from density_prediction_training import N_BINS

def load_ground_truth_data():
    """Load the preprocessed ground truth densities"""
    with open("ground_truth_histograms.pkl", 'rb') as f:
        return pickle.load(f)

def find_camera_ground_truth(camera_path: str, ground_truth_data: List[Dict]) -> Dict:
    """Find the ground truth data for a specific camera"""
    for gt_data in ground_truth_data:
        if gt_data['file_path'] == camera_path:
            return gt_data
    return None

def evaluate_camera_at_percentage_improved(model, camera_data: pd.DataFrame, 
                                         camera_gt: Dict, percentage: float, 
                                         n_trials: int = 10) -> Dict:
    """
    Evaluate model performance using preprocessed ground truth densities
    
    Args:
        model: Trained model
        camera_data: Camera DataFrame
        camera_gt: Ground truth densities for this camera
        percentage: Percentage of data to use
        n_trials: Number of random trials
    
    Returns:
        Dictionary with performance metrics
    """
    sample_size = max(1, int(len(camera_data) * percentage))
    
    # Multiple trials for stable estimates
    r2_scores = []
    tp_r2_scores = []
    fp_r2_scores = []
    
    # Ground truth densities (already normalized)
    gt_tp = camera_gt['tp_density']
    gt_fp = camera_gt['fp_density']
    
    for trial in range(n_trials):
        try:
            # Sample data
            if sample_size >= len(camera_data):
                sample_data = camera_data
            else:
                sample_data = camera_data.sample(n=sample_size, replace=False)
            
            # Predict densities
            pred_tp, pred_fp = predict_densities(model, sample_data, sample_size=len(sample_data))
            
            # Calculate R² scores against preprocessed ground truth
            tp_r2 = r2_score(gt_tp, pred_tp)
            fp_r2 = r2_score(gt_fp, pred_fp)
            
            tp_r2_scores.append(max(0, tp_r2))  # Clip negative R² to 0
            fp_r2_scores.append(max(0, fp_r2))
            
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
        'std_tp_r2': np.std(tp_r2_scores) if tp_r2_scores else 0.0,
        'std_fp_r2': np.std(fp_r2_scores) if fp_r2_scores else 0.0,
        'n_successful_trials': len(r2_scores)
    }

def evaluate_convergence_improved(model, ground_truth_data: List[Dict], percentages: np.ndarray, 
                                n_cameras: int = 50, n_trials_per_camera: int = 5) -> pd.DataFrame:
    """
    Evaluate model convergence using preprocessed ground truth data
    
    Args:
        model: Trained model
        ground_truth_data: List of preprocessed ground truth data
        percentages: Array of percentages to test
        n_cameras: Number of random cameras to evaluate
        n_trials_per_camera: Number of trials per camera per percentage
    
    Returns:
        DataFrame with convergence results
    """
    # Select random cameras from ground truth data
    selected_cameras = random.sample(ground_truth_data, min(n_cameras, len(ground_truth_data)))
    
    results = []
    
    print(f"Evaluating {len(selected_cameras)} cameras across {len(percentages)} percentages...")
    
    for camera_gt in tqdm(selected_cameras, desc="Processing cameras"):
        try:
            # Load camera data
            camera_data = pd.read_parquet(camera_gt['file_path'])
            
            # Skip cameras with insufficient data
            if len(camera_data) < 50:
                continue
                
            tp_count = camera_data['is_theft'].sum()
            fp_count = len(camera_data) - tp_count
            
            if tp_count < 5 or fp_count < 5:
                continue
            
            camera_name = Path(camera_gt['file_path']).stem
            
            # Evaluate at each percentage
            for percentage in percentages:
                metrics = evaluate_camera_at_percentage_improved(
                    model, camera_data, camera_gt, percentage, n_trials_per_camera
                )
                
                metrics['camera'] = camera_name
                metrics['camera_size'] = len(camera_data)
                results.append(metrics)
        
        except Exception as e:
            print(f"Error processing {camera_gt['file_path']}: {e}")
            continue
    
    return pd.DataFrame(results)

def plot_convergence_improved(results_df: pd.DataFrame, output_path: str = "convergence_analysis_improved.png"):
    """
    Create improved convergence plot
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate mean and std across cameras for each percentage
    convergence_stats = results_df.groupby('percentage').agg({
        'mean_r2': ['mean', 'std', 'count'],
        'mean_tp_r2': ['mean', 'std'],
        'mean_fp_r2': ['mean', 'std'],
        'std_r2': 'mean',  # Average std across cameras
        'sample_size': 'mean'
    }).reset_index()
    
    # Flatten column names
    convergence_stats.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                for col in convergence_stats.columns.values]
    
    percentages = convergence_stats['percentage'] * 100  # Convert to percentage
    
    # Main convergence plot with better error bars
    ax1 = axes[0, 0]
    mean_r2 = convergence_stats['mean_r2_mean']
    std_r2 = convergence_stats['mean_r2_std']
    
    ax1.plot(percentages, mean_r2, 'b-', linewidth=3, label='Mean R² Score', marker='o', markersize=6)
    ax1.fill_between(percentages, mean_r2 - std_r2, mean_r2 + std_r2, 
                     alpha=0.2, color='blue', label='±1 Std Dev (across cameras)')
    
    ax1.set_xlabel('Sample Size (% of Camera Data)', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Model Performance Convergence\n(Mean ± Std across cameras)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1)
    
    # Add text annotations for key points
    if len(mean_r2) > 0:
        max_r2_idx = np.argmax(mean_r2)
        ax1.annotate(f'Peak: {mean_r2.iloc[max_r2_idx]:.3f}\n@{percentages.iloc[max_r2_idx]:.0f}%', 
                    xy=(percentages.iloc[max_r2_idx], mean_r2.iloc[max_r2_idx]),
                    xytext=(percentages.iloc[max_r2_idx] + 15, mean_r2.iloc[max_r2_idx] + 0.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # TP vs FP convergence with improved styling
    ax2 = axes[0, 1]
    ax2.plot(percentages, convergence_stats['mean_tp_r2_mean'], 'r-', 
             linewidth=3, label='True Positive R²', marker='s', markersize=5)
    ax2.plot(percentages, convergence_stats['mean_fp_r2_mean'], 'g-', 
             linewidth=3, label='False Positive R²', marker='^', markersize=5)
    
    ax2.fill_between(percentages, 
                     convergence_stats['mean_tp_r2_mean'] - convergence_stats['mean_tp_r2_std'],
                     convergence_stats['mean_tp_r2_mean'] + convergence_stats['mean_tp_r2_std'],
                     alpha=0.15, color='red')
    ax2.fill_between(percentages,
                     convergence_stats['mean_fp_r2_mean'] - convergence_stats['mean_fp_r2_std'],
                     convergence_stats['mean_fp_r2_mean'] + convergence_stats['mean_fp_r2_std'],
                     alpha=0.15, color='green')
    
    ax2.set_xlabel('Sample Size (% of Camera Data)', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('TP vs FP Performance Convergence', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1)
    
    # Individual camera trajectories (sample)
    ax3 = axes[1, 0]
    sample_cameras = results_df['camera'].unique()[:8]  # Show 8 example cameras
    colors = plt.cm.tab10(np.linspace(0, 1, len(sample_cameras)))
    
    for i, camera in enumerate(sample_cameras):
        camera_data = results_df[results_df['camera'] == camera]
        camera_data = camera_data.sort_values('percentage')
        ax3.plot(camera_data['percentage'] * 100, camera_data['mean_r2'], 
                color=colors[i], alpha=0.7, linewidth=1.5, marker='o', markersize=3,
                label=f'Camera {i+1}')
    
    ax3.set_xlabel('Sample Size (% of Camera Data)', fontsize=12)
    ax3.set_ylabel('R² Score', fontsize=12)
    ax3.set_title('Individual Camera Convergence Trajectories\n(Sample of cameras)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 1)
    
    # Convergence efficiency analysis
    ax4 = axes[1, 1]
    
    # Calculate efficiency: R² improvement per additional % of data
    if len(convergence_stats) > 1:
        efficiency = np.diff(convergence_stats['mean_r2_mean']) / np.diff(convergence_stats['percentage'])
        actual_percentages = convergence_stats['percentage'] * 100
        pct_midpoints = (actual_percentages.iloc[1:].values + actual_percentages.iloc[:-1].values) / 2
        
        ax4.plot(pct_midpoints, efficiency, 'purple', linewidth=3, marker='d', markersize=6)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Find the point where efficiency drops below a threshold
        efficiency_threshold = 0.001  # 0.1% R² improvement per % data
        efficient_region = pct_midpoints[efficiency > efficiency_threshold]
        if len(efficient_region) > 0:
            ax4.axvline(x=efficient_region[-1], color='red', linestyle=':', alpha=0.7, 
                       label=f'Efficiency threshold\n({efficient_region[-1]:.0f}%)')
            ax4.legend(fontsize=10)
    
    ax4.set_xlabel('Sample Size (% of Camera Data)', fontsize=12)
    ax4.set_ylabel('Efficiency (ΔR² / Δ%)', fontsize=12)
    ax4.set_title('Convergence Efficiency\n(Diminishing returns analysis)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Improved convergence plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate model convergence with preprocessed ground truth')
    parser.add_argument('--n_cameras', type=int, default=100, 
                       help='Number of random cameras to evaluate')
    parser.add_argument('--n_trials', type=int, default=5,
                       help='Number of trials per camera per percentage')
    parser.add_argument('--output', type=str, default='convergence_analysis_improved.png',
                       help='Output plot filename')
    parser.add_argument('--save_results', type=str, default='convergence_results_improved.csv',
                       help='Save detailed results to CSV')
    
    args = parser.parse_args()
    
    # Load the best model
    print("Loading best model...")
    model, info = load_best_model()
    
    # Load ground truth data
    print("Loading preprocessed ground truth data...")
    ground_truth_data = load_ground_truth_data()
    print(f"Found {len(ground_truth_data)} cameras with ground truth data")
    
    if len(ground_truth_data) == 0:
        print("Error: No ground truth data found")
        return
    
    # Define percentage range (more granular at low percentages)
    percentages = np.array([0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 
                           0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # Evaluate convergence
    results_df = evaluate_convergence_improved(
        model, ground_truth_data, percentages, 
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
    plot_convergence_improved(results_df, args.output)
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("IMPROVED CONVERGENCE ANALYSIS SUMMARY")
    print("="*80)
    
    summary = results_df.groupby('percentage').agg({
        'mean_r2': ['mean', 'std', 'min', 'max'],
        'mean_tp_r2': 'mean',
        'mean_fp_r2': 'mean',
        'sample_size': 'mean'
    }).round(4)
    
    print("\nDetailed Performance by Sample Size:")
    print(summary)
    
    # Find optimal sample sizes
    convergence_stats = results_df.groupby('percentage')['mean_r2'].mean()
    
    # Peak performance
    peak_idx = convergence_stats.idxmax()
    peak_performance = convergence_stats.max()
    
    # 95% of peak performance
    target_performance = 0.95 * peak_performance
    optimal_idx = convergence_stats[convergence_stats >= target_performance].index[0]
    
    print(f"\n📊 KEY FINDINGS:")
    print(f"   Peak Performance: {peak_performance:.4f} R² at {peak_idx*100:.0f}% of data")
    print(f"   95% of Peak: {target_performance:.4f} R² achieved at {optimal_idx*100:.0f}% of data")
    print(f"   Efficiency Gain: {(peak_idx - optimal_idx) / peak_idx * 100:.1f}% less data needed")
    
    # Calculate convergence rate
    early_improvement = convergence_stats.iloc[2] - convergence_stats.iloc[0]  # 5% vs 1%
    late_improvement = convergence_stats.iloc[-1] - convergence_stats.iloc[-3]  # 100% vs 80%
    
    print(f"\n🎯 CONVERGENCE CHARACTERISTICS:")
    print(f"   Early improvement (1%→5%): {early_improvement:.4f} R²")
    print(f"   Late improvement (80%→100%): {late_improvement:.4f} R²")
    print(f"   Early/Late ratio: {early_improvement/late_improvement:.1f}x faster early convergence")

if __name__ == "__main__":
    main() 