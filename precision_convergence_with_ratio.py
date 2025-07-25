#!/usr/bin/env python3
"""
Precision Convergence Analysis with TP Ratio Prediction
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
from load_best_model_with_ratio import (
    load_best_model_with_ratio, 
    predict_densities_and_ratio,
    calculate_precision_from_predictions
)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def calculate_ground_truth_precision(camera_data):
    """Calculate ground truth precision curve for a camera"""
    
    # Create probability bins
    bins = np.linspace(0, 1, 21)  # 20 bins
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    precision_values = []
    
    for i in range(len(bins) - 1):
        # Get alerts in this probability bin
        mask = (camera_data['max_proba'] >= bins[i]) & (camera_data['max_proba'] < bins[i+1])
        bin_data = camera_data[mask]
        
        if len(bin_data) == 0:
            precision_values.append(0.0)  # No data in this bin
        else:
            tp_count = len(bin_data[bin_data['is_theft'] == 1])
            total_count = len(bin_data)
            precision = tp_count / total_count
            precision_values.append(precision)
    
    return np.array(precision_values)

def calculate_precision_error(predicted_precision, ground_truth_precision):
    """Calculate mean absolute error between predicted and ground truth precision"""
    return np.mean(np.abs(predicted_precision - ground_truth_precision))

def evaluate_precision_convergence_with_ratio(n_cameras=50, sample_sizes=None):
    """
    Evaluate precision error as a function of sample size per camera using TP ratio model
    
    Args:
        n_cameras: Number of cameras to evaluate
        sample_sizes: List of sample sizes to test (if None, use adaptive range)
    """
    
    print("🎯 Precision Convergence Analysis with TP Ratio")
    print("=" * 60)
    
    # Load improved model with ratio
    print("Loading model with TP ratio prediction...")
    model, model_info = load_best_model_with_ratio()
    print(f"Model R² Score: {model_info['r2_score']:.4f}")
    print(f"Ratio R² Score: {model_info['r2_ratio']:.4f}")
    
    # Load ground truth data
    print("Loading ground truth data...")
    with open('ground_truth_histograms.pkl', 'rb') as f:
        gt_data = pickle.load(f)
    
    # Select diverse cameras for evaluation
    print(f"Selecting {n_cameras} cameras for evaluation...")
    
    # Get camera sizes and calculate ground truth precision curves
    camera_info = []
    for gt in tqdm(gt_data, desc="Processing cameras"):
        try:
            df = pd.read_parquet(gt['file_path'])
            if len(df) < 100:  # Skip cameras with too few alerts
                continue
                
            gt_precision = calculate_ground_truth_precision(df)
            camera_info.append({
                'size': len(df),
                'data': df,
                'gt_precision': gt_precision,
                'file_path': gt['file_path']
            })
        except:
            continue
    
    # Sort by size and select diverse range
    camera_info.sort(key=lambda x: x['size'])
    
    # Select cameras across size spectrum
    selected_cameras = []
    n_per_quartile = n_cameras // 4
    
    # Small cameras (0-25th percentile)
    q1_end = len(camera_info) // 4
    selected_cameras.extend(random.sample(camera_info[:q1_end], min(n_per_quartile, q1_end)))
    
    # Medium-small cameras (25-50th percentile)
    q2_start, q2_end = q1_end, len(camera_info) // 2
    selected_cameras.extend(random.sample(camera_info[q2_start:q2_end], min(n_per_quartile, q2_end-q2_start)))
    
    # Medium-large cameras (50-75th percentile)
    q3_start, q3_end = q2_end, 3 * len(camera_info) // 4
    selected_cameras.extend(random.sample(camera_info[q3_start:q3_end], min(n_per_quartile, q3_end-q3_start)))
    
    # Large cameras (75-100th percentile)
    q4_start = q3_end
    remaining = n_cameras - len(selected_cameras)
    selected_cameras.extend(random.sample(camera_info[q4_start:], min(remaining, len(camera_info)-q4_start)))
    
    print(f"Selected {len(selected_cameras)} cameras")
    print(f"Camera sizes range: {min(c['size'] for c in selected_cameras)} - {max(c['size'] for c in selected_cameras)} alerts")
    
    # Define sample sizes to test
    if sample_sizes is None:
        min_size = min(c['size'] for c in selected_cameras)
        max_size = max(c['size'] for c in selected_cameras)
        
        # Create adaptive sample size range
        sample_sizes = []
        
        # Small sizes (detailed resolution)
        sample_sizes.extend(list(range(10, 50, 5)))
        
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
        
        for camera in selected_cameras:
            try:
                # Skip if sample size is too large for this camera
                if sample_size > camera['size']:
                    continue
                
                # Sample data from camera
                sample_data = camera['data'].sample(n=sample_size, replace=False, random_state=42)
                
                # Predict using model with TP ratio
                tp_density, fp_density, tp_ratio = predict_densities_and_ratio(
                    model, sample_data, sample_size=sample_size
                )
                
                # Calculate predicted precision curve
                predicted_precision = calculate_precision_from_predictions(
                    tp_density, fp_density, tp_ratio
                )
                
                # Calculate precision error
                precision_error = calculate_precision_error(
                    predicted_precision, camera['gt_precision']
                )
                
                size_results.append({
                    'camera_size': camera['size'],
                    'sample_size': sample_size,
                    'precision_error': precision_error,
                    'predicted_tp_ratio': tp_ratio,
                    'actual_tp_ratio': len(sample_data[sample_data['is_theft'] == 1]) / len(sample_data)
                })
                
            except Exception as e:
                continue
        
        if size_results:
            # Calculate statistics for this sample size
            precision_errors = [r['precision_error'] for r in size_results]
            tp_ratio_errors = [abs(r['predicted_tp_ratio'] - r['actual_tp_ratio']) for r in size_results]
            
            results.append({
                'sample_size': sample_size,
                'mean_precision_error': np.mean(precision_errors),
                'std_precision_error': np.std(precision_errors),
                'median_precision_error': np.median(precision_errors),
                'min_precision_error': np.min(precision_errors),
                'max_precision_error': np.max(precision_errors),
                'mean_tp_ratio_error': np.mean(tp_ratio_errors),
                'std_tp_ratio_error': np.std(tp_ratio_errors),
                'n_cameras': len(size_results)
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('precision_convergence_with_ratio_results.csv', index=False)
    print(f"\n✅ Results saved to precision_convergence_with_ratio_results.csv")
    
    # Create precision convergence plot
    create_precision_plot_with_ratio(results_df)
    
    # Print summary statistics
    print_precision_summary_with_ratio(results_df)
    
    return results_df

def create_precision_plot_with_ratio(results_df):
    """Create precision error convergence plot with TP ratio analysis"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    x = results_df['sample_size']
    
    # Plot 1: Precision Error vs Sample Size
    y1 = results_df['mean_precision_error']
    yerr1 = results_df['std_precision_error']
    
    ax1.plot(x, y1, 'o-', linewidth=2, markersize=6, label='Mean Precision Error', color='red', alpha=0.8)
    ax1.fill_between(x, y1 - yerr1, y1 + yerr1, alpha=0.3, color='lightcoral', label='±1 Std Dev')
    
    # Add median line
    ax1.plot(x, results_df['median_precision_error'], 's--', linewidth=1.5, markersize=4, 
            label='Median Precision Error', color='darkred', alpha=0.7)
    
    # Add target precision lines
    ax1.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='5% Error Target')
    ax1.axhline(y=0.10, color='orange', linestyle='--', alpha=0.7, label='10% Error Target')
    ax1.axhline(y=0.15, color='red', linestyle='--', alpha=0.7, label='15% Error Threshold')
    
    ax1.set_xlabel('Sample Size (alerts per camera)', fontsize=12)
    ax1.set_ylabel('Average Precision Error', fontsize=12)
    ax1.set_title('🎯 Precision Error vs Sample Size\nUsing Model with TP Ratio Prediction', 
                fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Set log scale for x-axis if range is large
    if max(x) > 10 * min(x):
        ax1.set_xscale('log')
        ax1.set_xlabel('Sample Size (alerts per camera) - Log Scale', fontsize=12)
    
    # Plot 2: TP Ratio Error vs Sample Size
    y2 = results_df['mean_tp_ratio_error']
    yerr2 = results_df['std_tp_ratio_error']
    
    ax2.plot(x, y2, 'o-', linewidth=2, markersize=6, label='Mean TP Ratio Error', color='blue', alpha=0.8)
    ax2.fill_between(x, y2 - yerr2, y2 + yerr2, alpha=0.3, color='lightblue', label='±1 Std Dev')
    
    # Add target lines for TP ratio
    ax2.axhline(y=0.02, color='green', linestyle='--', alpha=0.7, label='2% Ratio Error Target')
    ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='5% Ratio Error Target')
    ax2.axhline(y=0.10, color='red', linestyle='--', alpha=0.7, label='10% Ratio Error Threshold')
    
    ax2.set_xlabel('Sample Size (alerts per camera)', fontsize=12)
    ax2.set_ylabel('Average TP Ratio Error', fontsize=12)
    ax2.set_title('📊 TP Ratio Prediction Error vs Sample Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Set log scale for x-axis if range is large
    if max(x) > 10 * min(x):
        ax2.set_xscale('log')
        ax2.set_xlabel('Sample Size (alerts per camera) - Log Scale', fontsize=12)
    
    # Add annotations for key points
    min_error_idx = results_df['mean_precision_error'].idxmin()
    min_error_size = results_df.iloc[min_error_idx]['sample_size']
    min_error_val = results_df.iloc[min_error_idx]['mean_precision_error']
    
    ax1.annotate(f'Best: {min_error_val:.3f} error\nat {min_error_size} samples',
                xy=(min_error_size, min_error_val),
                xytext=(min_error_size * 2, min_error_val + 0.02),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, ha='left',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('precision_convergence_with_ratio_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Plot saved to precision_convergence_with_ratio_plot.png")

def print_precision_summary_with_ratio(results_df):
    """Print precision convergence summary statistics with TP ratio analysis"""
    
    print("\n" + "="*70)
    print("📈 PRECISION CONVERGENCE SUMMARY (WITH TP RATIO)")
    print("="*70)
    
    # Overall statistics
    best_precision = results_df['mean_precision_error'].min()
    worst_precision = results_df['mean_precision_error'].max()
    best_size = results_df.loc[results_df['mean_precision_error'].idxmin(), 'sample_size']
    worst_size = results_df.loc[results_df['mean_precision_error'].idxmax(), 'sample_size']
    
    best_ratio = results_df['mean_tp_ratio_error'].min()
    worst_ratio = results_df['mean_tp_ratio_error'].max()
    
    print(f"\n🎯 PRECISION PERFORMANCE:")
    print(f"   Best Error:       {best_precision:.4f} (at {best_size} samples)")
    print(f"   Worst Error:      {worst_precision:.4f} (at {worst_size} samples)")
    print(f"   Improvement:      {worst_precision - best_precision:.4f} reduction")
    print(f"   Relative Gain:    {((worst_precision - best_precision) / worst_precision * 100):.1f}%")
    
    print(f"\n📊 TP RATIO PERFORMANCE:")
    print(f"   Best Ratio Error: {best_ratio:.4f}")
    print(f"   Worst Ratio Error:{worst_ratio:.4f}")
    print(f"   Average Ratio Error: {results_df['mean_tp_ratio_error'].mean():.4f}")
    
    # Sample size analysis
    print(f"\n📊 SAMPLE SIZE ANALYSIS:")
    small_samples = results_df[results_df['sample_size'] <= 50]
    medium_samples = results_df[(results_df['sample_size'] > 50) & (results_df['sample_size'] <= 200)]
    large_samples = results_df[results_df['sample_size'] > 200]
    
    if not small_samples.empty:
        print(f"   Small (≤50):      {small_samples['mean_precision_error'].mean():.4f} precision error, {small_samples['mean_tp_ratio_error'].mean():.4f} ratio error")
    if not medium_samples.empty:
        print(f"   Medium (50-200):  {medium_samples['mean_precision_error'].mean():.4f} precision error, {medium_samples['mean_tp_ratio_error'].mean():.4f} ratio error")
    if not large_samples.empty:
        print(f"   Large (>200):     {large_samples['mean_precision_error'].mean():.4f} precision error, {large_samples['mean_tp_ratio_error'].mean():.4f} ratio error")
    
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
        
        first_avg_prec = first_half['mean_precision_error'].mean()
        second_avg_prec = second_half['mean_precision_error'].mean()
        prec_improvement = first_avg_prec - second_avg_prec
        
        first_avg_ratio = first_half['mean_tp_ratio_error'].mean()
        second_avg_ratio = second_half['mean_tp_ratio_error'].mean()
        ratio_improvement = first_avg_ratio - second_avg_ratio
        
        print(f"   Early samples:    {first_avg_prec:.4f} precision error, {first_avg_ratio:.4f} ratio error")
        print(f"   Later samples:    {second_avg_prec:.4f} precision error, {second_avg_ratio:.4f} ratio error")
        print(f"   Precision Improv: {prec_improvement:.4f} ({prec_improvement/first_avg_prec*100:.1f}%)")
        print(f"   Ratio Improv:     {ratio_improvement:.4f} ({ratio_improvement/first_avg_ratio*100:.1f}%)")
        
        if prec_improvement > 0:
            print(f"   ✅ Clear convergence: both precision and ratio improve with more data")
        else:
            print(f"   ⚠️  Mixed results: check individual metric convergence")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    # Run precision convergence analysis with TP ratio
    results = evaluate_precision_convergence_with_ratio(
        n_cameras=60,  # Good sample size
        sample_sizes=None  # Auto-generate adaptive range
    )
    
    print("\n🎉 Precision convergence analysis with TP ratio complete!")
    print("📁 Check precision_convergence_with_ratio_results.csv and precision_convergence_with_ratio_plot.png") 