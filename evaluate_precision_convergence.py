#!/usr/bin/env python3
"""
Precision Convergence Analysis
Evaluates average precision error as a function of sample size per camera
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from tqdm import tqdm
import random
from load_best_model_improved import load_best_model_improved, predict_densities_improved

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def calculate_precision_error(predicted_tp, predicted_fp, ground_truth_tp, ground_truth_fp):
    """
    Calculate precision error between predicted and ground truth densities
    
    Args:
        predicted_tp: Predicted TP density (50 bins)
        predicted_fp: Predicted FP density (50 bins)
        ground_truth_tp: Ground truth TP density (50 bins)
        ground_truth_fp: Ground truth FP density (50 bins)
    
    Returns:
        float: Average absolute precision error
    """
    # Calculate absolute errors for each bin
    tp_errors = np.abs(predicted_tp - ground_truth_tp)
    fp_errors = np.abs(predicted_fp - ground_truth_fp)
    
    # Average error across all bins
    avg_error = np.mean(np.concatenate([tp_errors, fp_errors]))
    
    return avg_error

def evaluate_precision_convergence(n_cameras=50, sample_sizes=None):
    """
    Evaluate precision error as a function of sample size per camera
    
    Args:
        n_cameras: Number of cameras to evaluate
        sample_sizes: List of sample sizes to test (if None, use adaptive range)
    """
    
    print("🎯 Precision Convergence Analysis")
    print("=" * 50)
    
    # Load improved model
    print("Loading improved model...")
    model, model_info = load_best_model_improved()
    print(f"Model R² Score: {model_info['r2_score']:.4f}")
    
    # Load ground truth data
    print("Loading ground truth data...")
    with open('ground_truth_histograms.pkl', 'rb') as f:
        gt_data = pickle.load(f)
    
    # Select diverse cameras for evaluation
    print(f"Selecting {n_cameras} cameras for evaluation...")
    
    # Get camera sizes for diverse selection
    camera_sizes = []
    for gt in gt_data:
        try:
            df = pd.read_parquet(gt['file_path'])
            camera_sizes.append((len(df), gt))
        except:
            continue
    
    # Sort by size and select diverse range
    camera_sizes.sort(key=lambda x: x[0])
    
    # Select cameras across size spectrum
    selected_cameras = []
    n_per_quartile = n_cameras // 4
    
    # Small cameras (0-25th percentile)
    q1_end = len(camera_sizes) // 4
    selected_cameras.extend(random.sample(camera_sizes[:q1_end], min(n_per_quartile, q1_end)))
    
    # Medium-small cameras (25-50th percentile)
    q2_start, q2_end = q1_end, len(camera_sizes) // 2
    selected_cameras.extend(random.sample(camera_sizes[q2_start:q2_end], min(n_per_quartile, q2_end-q2_start)))
    
    # Medium-large cameras (50-75th percentile)
    q3_start, q3_end = q2_end, 3 * len(camera_sizes) // 4
    selected_cameras.extend(random.sample(camera_sizes[q3_start:q3_end], min(n_per_quartile, q3_end-q3_start)))
    
    # Large cameras (75-100th percentile)
    q4_start = q3_end
    remaining = n_cameras - len(selected_cameras)
    selected_cameras.extend(random.sample(camera_sizes[q4_start:], min(remaining, len(camera_sizes)-q4_start)))
    
    print(f"Selected {len(selected_cameras)} cameras")
    print(f"Camera sizes range: {min(s[0] for s in selected_cameras)} - {max(s[0] for s in selected_cameras)} alerts")
    
    # Define sample sizes to test
    if sample_sizes is None:
        min_size = min(s[0] for s in selected_cameras)
        max_size = max(s[0] for s in selected_cameras)
        
        # Create adaptive sample size range
        sample_sizes = []
        
        # Small sizes (detailed resolution)
        sample_sizes.extend(list(range(5, 50, 5)))
        
        # Medium sizes (moderate resolution)
        sample_sizes.extend(list(range(50, 200, 10)))
        
        # Large sizes (coarse resolution)
        sample_sizes.extend(list(range(200, min(1000, max_size), 50)))
        
        # Very large sizes if needed
        if max_size > 1000:
            sample_sizes.extend(list(range(1000, max_size, 100)))
        
        sample_sizes = sorted(list(set(sample_sizes)))
    
    print(f"Testing {len(sample_sizes)} sample sizes from {min(sample_sizes)} to {max(sample_sizes)}")
    
    results = []
    
    print(f"\nEvaluating precision error vs sample size...")
    
    for sample_size in tqdm(sample_sizes, desc="Sample Size"):
        size_results = []
        
        for camera_size, gt_camera in selected_cameras:
            try:
                # Skip if sample size is too large for this camera
                if sample_size > camera_size:
                    continue
                
                # Load camera data
                camera_data = pd.read_parquet(gt_camera['file_path'])
                
                # Sample data
                sample_data = camera_data.sample(n=sample_size, replace=False, random_state=42)
                
                # Predict using improved model
                pred_tp, pred_fp = predict_densities_improved(model, sample_data, sample_size=len(sample_data))
                
                # Calculate precision error
                precision_error = calculate_precision_error(
                    pred_tp, pred_fp,
                    gt_camera['tp_density'], gt_camera['fp_density']
                )
                
                size_results.append({
                    'camera_size': camera_size,
                    'sample_size': sample_size,
                    'precision_error': precision_error
                })
                
            except Exception as e:
                continue
        
        if size_results:
            # Calculate statistics for this sample size
            precision_errors = [r['precision_error'] for r in size_results]
            
            results.append({
                'sample_size': sample_size,
                'mean_precision_error': np.mean(precision_errors),
                'std_precision_error': np.std(precision_errors),
                'median_precision_error': np.median(precision_errors),
                'min_precision_error': np.min(precision_errors),
                'max_precision_error': np.max(precision_errors),
                'n_cameras': len(size_results)
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('precision_convergence_results.csv', index=False)
    print(f"\n✅ Results saved to precision_convergence_results.csv")
    
    # Create precision convergence plot
    create_precision_plot(results_df)
    
    # Print summary statistics
    print_precision_summary(results_df)
    
    return results_df

def create_precision_plot(results_df):
    """Create precision error convergence plot"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    x = results_df['sample_size']
    y = results_df['mean_precision_error']
    yerr = results_df['std_precision_error']
    
    # Main plot: precision error vs sample size
    ax.plot(x, y, 'o-', linewidth=2, markersize=6, label='Mean Precision Error', color='red', alpha=0.8)
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.3, color='lightcoral', label='±1 Std Dev')
    
    # Add median line
    ax.plot(x, results_df['median_precision_error'], 's--', linewidth=1.5, markersize=4, 
            label='Median Precision Error', color='darkred', alpha=0.7)
    
    # Add target precision lines
    ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='5% Error Target')
    ax.axhline(y=0.10, color='orange', linestyle='--', alpha=0.7, label='10% Error Target')
    ax.axhline(y=0.15, color='red', linestyle='--', alpha=0.7, label='15% Error Threshold')
    
    ax.set_xlabel('Sample Size (alerts per camera)', fontsize=12)
    ax.set_ylabel('Average Precision Error', fontsize=12)
    ax.set_title('🎯 Precision Error vs Sample Size\nHow prediction accuracy improves with more data', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set log scale for x-axis if range is large
    if max(x) > 10 * min(x):
        ax.set_xscale('log')
        ax.set_xlabel('Sample Size (alerts per camera) - Log Scale', fontsize=12)
    
    # Format y-axis as percentage
    ax.set_ylim(0, max(y + yerr) * 1.1)
    
    # Add annotations for key points
    min_error_idx = results_df['mean_precision_error'].idxmin()
    min_error_size = results_df.iloc[min_error_idx]['sample_size']
    min_error_val = results_df.iloc[min_error_idx]['mean_precision_error']
    
    ax.annotate(f'Best: {min_error_val:.3f} error\nat {min_error_size} samples',
                xy=(min_error_size, min_error_val),
                xytext=(min_error_size * 2, min_error_val + 0.02),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, ha='left',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('precision_convergence_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Plot saved to precision_convergence_plot.png")

def print_precision_summary(results_df):
    """Print precision convergence summary statistics"""
    
    print("\n" + "="*60)
    print("📈 PRECISION CONVERGENCE SUMMARY")
    print("="*60)
    
    # Overall statistics
    best_performance = results_df['mean_precision_error'].min()
    worst_performance = results_df['mean_precision_error'].max()
    best_size = results_df.loc[results_df['mean_precision_error'].idxmin(), 'sample_size']
    worst_size = results_df.loc[results_df['mean_precision_error'].idxmax(), 'sample_size']
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   Best Error:       {best_performance:.4f} (at {best_size} samples)")
    print(f"   Worst Error:      {worst_performance:.4f} (at {worst_size} samples)")
    print(f"   Improvement:      {worst_performance - best_performance:.4f} reduction")
    print(f"   Relative Gain:    {((worst_performance - best_performance) / worst_performance * 100):.1f}%")
    
    # Sample size analysis
    print(f"\n📊 SAMPLE SIZE ANALYSIS:")
    small_samples = results_df[results_df['sample_size'] <= 50]
    medium_samples = results_df[(results_df['sample_size'] > 50) & (results_df['sample_size'] <= 200)]
    large_samples = results_df[results_df['sample_size'] > 200]
    
    if not small_samples.empty:
        print(f"   Small (≤50):      {small_samples['mean_precision_error'].mean():.4f} avg error")
    if not medium_samples.empty:
        print(f"   Medium (50-200):  {medium_samples['mean_precision_error'].mean():.4f} avg error")
    if not large_samples.empty:
        print(f"   Large (>200):     {large_samples['mean_precision_error'].mean():.4f} avg error")
    
    # Precision targets
    print(f"\n🏆 PRECISION TARGETS:")
    targets = [0.05, 0.10, 0.15, 0.20]
    for target in targets:
        target_reached = results_df[results_df['mean_precision_error'] <= target]
        if not target_reached.empty:
            min_samples = target_reached['sample_size'].min()
            print(f"   {target:.0%} Error:      {min_samples:4.0f} samples needed")
        else:
            print(f"   {target:.0%} Error:      Not achieved")
    
    # Convergence analysis
    print(f"\n📈 CONVERGENCE PATTERN:")
    
    # Calculate improvement rate
    if len(results_df) >= 2:
        first_half = results_df.head(len(results_df)//2)
        second_half = results_df.tail(len(results_df)//2)
        
        first_avg = first_half['mean_precision_error'].mean()
        second_avg = second_half['mean_precision_error'].mean()
        improvement = first_avg - second_avg
        
        print(f"   Early samples:    {first_avg:.4f} avg error")
        print(f"   Later samples:    {second_avg:.4f} avg error")
        print(f"   Improvement:      {improvement:.4f} ({improvement/first_avg*100:.1f}%)")
        
        if improvement > 0:
            print(f"   ✅ Clear convergence: precision improves with more data")
        else:
            print(f"   ⚠️  Plateau reached: no significant improvement with more data")
    
    # Stability analysis
    print(f"\n🎚️  STABILITY ANALYSIS:")
    stability = results_df['std_precision_error'].mean()
    print(f"   Mean Std Dev:     {stability:.4f}")
    print(f"   Coefficient Var:  {(stability / results_df['mean_precision_error'].mean()):.2f}")
    
    if stability < 0.02:
        print(f"   ✅ Very stable across cameras")
    elif stability < 0.05:
        print(f"   ✅ Good stability across cameras")
    else:
        print(f"   ⚠️  Some variability across cameras")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Run precision convergence analysis
    results = evaluate_precision_convergence(
        n_cameras=60,  # Good sample size
        sample_sizes=None  # Auto-generate adaptive range
    )
    
    print("\n🎉 Precision convergence analysis complete!")
    print("📁 Check precision_convergence_results.csv and precision_convergence_plot.png") 