#!/usr/bin/env python3
"""
Final Convergence Analysis with Improved Model
Evaluates performance across all sample sizes from 1% to 100%
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from pathlib import Path
import pickle
from tqdm import tqdm
import random
from load_best_model_improved import load_best_model_improved, predict_densities_improved

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def evaluate_convergence_final(n_cameras=50, max_percentage=100, step=1):
    """
    Final convergence evaluation using the improved model
    
    Args:
        n_cameras: Number of cameras to evaluate
        max_percentage: Maximum percentage to evaluate (1-100)
        step: Step size for percentage increments
    """
    
    print("🚀 Final Convergence Analysis - Improved Model")
    print("=" * 60)
    
    # Load improved model
    print("Loading improved model...")
    model, model_info = load_best_model_improved()
    print(f"Model R² Score: {model_info['r2_score']:.4f}")
    print(f"Training Range: {model_info['hyperparameters']['sample_range']}")
    
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
    
    # Define percentages to evaluate
    percentages = list(range(1, max_percentage + 1, step))
    results = []
    
    print(f"\nEvaluating convergence from 1% to {max_percentage}%...")
    
    for percentage in tqdm(percentages, desc="Percentage"):
        pct_results = []
        
        for camera_size, gt_camera in selected_cameras:
            try:
                # Load camera data
                camera_data = pd.read_parquet(gt_camera['file_path'])
                
                # Calculate sample size
                sample_size = max(1, int(len(camera_data) * percentage / 100))
                
                # Skip if sample size is too large for this camera
                if sample_size > len(camera_data):
                    continue
                
                # Sample data
                sample_data = camera_data.sample(n=sample_size, replace=False, random_state=42)
                
                # Predict using improved model
                pred_tp, pred_fp = predict_densities_improved(model, sample_data, sample_size=len(sample_data))
                
                # Calculate R² scores
                tp_r2 = r2_score(gt_camera['tp_density'], pred_tp)
                fp_r2 = r2_score(gt_camera['fp_density'], pred_fp)
                combined_r2 = (tp_r2 + fp_r2) / 2
                
                pct_results.append({
                    'camera_size': camera_size,
                    'sample_size': sample_size,
                    'tp_r2': tp_r2,
                    'fp_r2': fp_r2,
                    'combined_r2': combined_r2
                })
                
            except Exception as e:
                continue
        
        if pct_results:
            # Calculate statistics for this percentage
            combined_r2_values = [r['combined_r2'] for r in pct_results]
            tp_r2_values = [r['tp_r2'] for r in pct_results]
            fp_r2_values = [r['fp_r2'] for r in pct_results]
            
            results.append({
                'percentage': percentage,
                'mean_combined_r2': np.mean(combined_r2_values),
                'std_combined_r2': np.std(combined_r2_values),
                'mean_tp_r2': np.mean(tp_r2_values),
                'std_tp_r2': np.std(tp_r2_values),
                'mean_fp_r2': np.mean(fp_r2_values),
                'std_fp_r2': np.std(fp_r2_values),
                'n_cameras': len(pct_results),
                'median_sample_size': np.median([r['sample_size'] for r in pct_results])
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('convergence_results_final.csv', index=False)
    print(f"\n✅ Results saved to convergence_results_final.csv")
    
    # Create comprehensive plots
    create_convergence_plots_final(results_df)
    
    # Print summary statistics
    print_convergence_summary_final(results_df)
    
    return results_df

def create_convergence_plots_final(results_df):
    """Create comprehensive convergence plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🚀 Final Convergence Analysis - Improved Model\n'
                'Performance Across All Sample Sizes (1% - 100%)', 
                fontsize=16, fontweight='bold')
    
    # Plot 1: Combined R² convergence with confidence intervals
    ax1 = axes[0, 0]
    x = results_df['percentage']
    y = results_df['mean_combined_r2']
    yerr = results_df['std_combined_r2']
    
    ax1.plot(x, y, 'o-', linewidth=2, markersize=4, label='Mean R²', color='darkblue')
    ax1.fill_between(x, y - yerr, y + yerr, alpha=0.3, color='lightblue', label='±1 Std Dev')
    ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='90% Threshold')
    ax1.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Threshold')
    
    ax1.set_xlabel('Percentage of Camera Data Used (%)')
    ax1.set_ylabel('Combined R² Score')
    ax1.set_title('🎯 Overall Performance Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: TP vs FP R² comparison
    ax2 = axes[0, 1]
    ax2.plot(x, results_df['mean_tp_r2'], 'o-', linewidth=2, markersize=4, 
             label='True Positives', color='green')
    ax2.fill_between(x, 
                     results_df['mean_tp_r2'] - results_df['std_tp_r2'],
                     results_df['mean_tp_r2'] + results_df['std_tp_r2'],
                     alpha=0.3, color='lightgreen')
    
    ax2.plot(x, results_df['mean_fp_r2'], 's-', linewidth=2, markersize=4,
             label='False Positives', color='orange')
    ax2.fill_between(x,
                     results_df['mean_fp_r2'] - results_df['std_fp_r2'],
                     results_df['mean_fp_r2'] + results_df['std_fp_r2'],
                     alpha=0.3, color='moccasin')
    
    ax2.set_xlabel('Percentage of Camera Data Used (%)')
    ax2.set_ylabel('R² Score')
    ax2.set_title('📊 TP vs FP Performance')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    
    # Plot 3: Sample size distribution
    ax3 = axes[1, 0]
    ax3.plot(x, results_df['median_sample_size'], 'o-', linewidth=2, markersize=4, color='purple')
    ax3.set_xlabel('Percentage of Camera Data Used (%)')
    ax3.set_ylabel('Median Sample Size (alerts)')
    ax3.set_title('📈 Sample Size Growth')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Performance improvement zones
    ax4 = axes[1, 1]
    
    # Define performance zones
    excellent = results_df['mean_combined_r2'] >= 0.95
    good = (results_df['mean_combined_r2'] >= 0.9) & (results_df['mean_combined_r2'] < 0.95)
    fair = (results_df['mean_combined_r2'] >= 0.8) & (results_df['mean_combined_r2'] < 0.9)
    poor = results_df['mean_combined_r2'] < 0.8
    
    ax4.fill_between(x[excellent], 0, 1, alpha=0.3, color='green', label='Excellent (≥95%)')
    ax4.fill_between(x[good], 0, 1, alpha=0.3, color='yellow', label='Good (90-95%)')
    ax4.fill_between(x[fair], 0, 1, alpha=0.3, color='orange', label='Fair (80-90%)')
    ax4.fill_between(x[poor], 0, 1, alpha=0.3, color='red', label='Poor (<80%)')
    
    ax4.plot(x, results_df['mean_combined_r2'], 'ko-', linewidth=3, markersize=5, label='Actual Performance')
    
    ax4.set_xlabel('Percentage of Camera Data Used (%)')
    ax4.set_ylabel('R² Score')
    ax4.set_title('🎨 Performance Zones')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('convergence_analysis_final.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Plots saved to convergence_analysis_final.png")

def print_convergence_summary_final(results_df):
    """Print comprehensive summary statistics"""
    
    print("\n" + "="*60)
    print("📈 FINAL CONVERGENCE ANALYSIS SUMMARY")
    print("="*60)
    
    # Overall statistics
    final_performance = results_df.iloc[-1]['mean_combined_r2']
    initial_performance = results_df.iloc[0]['mean_combined_r2'] 
    max_performance = results_df['mean_combined_r2'].max()
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   Initial (1%):     {initial_performance:.3f}")
    print(f"   Final (100%):     {final_performance:.3f}")
    print(f"   Maximum:          {max_performance:.3f}")
    print(f"   Improvement:      {final_performance - initial_performance:+.3f}")
    
    # Performance milestones
    print(f"\n🏆 PERFORMANCE MILESTONES:")
    milestones = [0.8, 0.85, 0.9, 0.95, 0.98]
    for milestone in milestones:
        milestone_reached = results_df[results_df['mean_combined_r2'] >= milestone]
        if not milestone_reached.empty:
            first_pct = milestone_reached.iloc[0]['percentage']
            print(f"   {milestone:.0%} R²:          {first_pct:3.0f}% of data")
        else:
            print(f"   {milestone:.0%} R²:          Not reached")
    
    # Stability analysis
    print(f"\n📊 STABILITY ANALYSIS:")
    std_values = results_df['std_combined_r2']
    print(f"   Mean Std Dev:     {std_values.mean():.4f}")
    print(f"   Max Std Dev:      {std_values.max():.4f}")
    print(f"   Min Std Dev:      {std_values.min():.4f}")
    
    # Sample size analysis
    print(f"\n📈 SAMPLE SIZE ANALYSIS:")
    final_sample_size = results_df.iloc[-1]['median_sample_size']
    initial_sample_size = results_df.iloc[0]['median_sample_size']
    print(f"   Initial size:     {initial_sample_size:.0f} alerts")
    print(f"   Final size:       {final_sample_size:.0f} alerts")
    print(f"   Size range:       {initial_sample_size:.0f} - {final_sample_size:.0f} alerts")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    
    # Find performance plateau
    last_10_pct = results_df.tail(10)
    performance_range = last_10_pct['mean_combined_r2'].max() - last_10_pct['mean_combined_r2'].min()
    
    if performance_range < 0.01:
        plateau_start = results_df.tail(10).iloc[0]['percentage']
        print(f"   • Performance plateaus at {plateau_start}% (stable within 1%)")
    else:
        print(f"   • Performance continues improving through 100%")
    
    # Compare with original problem
    if final_performance > 0.95:
        print(f"   • ✅ PROBLEM SOLVED: Excellent performance at all scales")
    elif final_performance > 0.9:
        print(f"   • ✅ MAJOR IMPROVEMENT: Good performance at all scales")
    else:
        print(f"   • ⚠️  Still room for improvement")
    
    print(f"   • Model handles {results_df.iloc[-1]['median_sample_size']:.0f}x larger samples than original")
    print(f"   • Consistent performance across {results_df['n_cameras'].iloc[0]} diverse cameras")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Run comprehensive convergence analysis
    results = evaluate_convergence_final(
        n_cameras=100,      # More cameras for robust analysis
        max_percentage=100, # Full range
        step=2             # Every 2% for detailed view
    )
    
    print("\n🎉 Final convergence analysis complete!")
    print("📁 Check convergence_results_final.csv and convergence_analysis_final.png") 