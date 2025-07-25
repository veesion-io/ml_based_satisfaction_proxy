#!/usr/bin/env python3
"""
Evaluate convergence performance of the precision-calibrated model
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm
from load_precision_calibrated_model import (
    load_precision_calibrated_model, 
    predict_average_precision_calibrated_model,
    compute_ground_truth_average_precision
)

def evaluate_convergence_precision_calibrated():
    """Evaluate how precision-calibrated model performance varies with sample size"""
    
    print("🔍 PRECISION-CALIBRATED MODEL CONVERGENCE ANALYSIS")
    print("=" * 55)
    
    # Load model
    model = load_precision_calibrated_model()
    
    # Load data
    print("📂 Loading camera data...")
    with open("ground_truth_histograms.pkl", 'rb') as f:
        data = pickle.load(f)
    
    # Use a subset for analysis
    num_cameras = min(200, len(data))
    test_cameras = np.random.choice(data, size=num_cameras, replace=False)
    
    print(f"📊 Analyzing {num_cameras} cameras...")
    
    # Define sample size percentages and absolute ranges
    percentages = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Store results
    results = {
        'percentage': [],
        'avg_sample_size': [],
        'r2_ap': [],
        'mean_error': [],
        'std_error': [],
        'num_valid_cameras': []
    }
    
    # Track per-camera results for individual trajectories
    camera_results = {}
    
    print("\n🧮 Evaluating across sample sizes...")
    
    for pct in tqdm(percentages, desc="Sample size percentages"):
        predictions = []
        ground_truths = []
        sample_sizes = []
        errors = []
        
        valid_cameras = 0
        
        for camera_data in test_cameras:
            try:
                # Load camera alerts
                alerts_df = pd.read_parquet(camera_data['file_path'])
                
                if len(alerts_df) < 50:  # Skip cameras with too few alerts
                    continue
                
                # Calculate sample size
                k = max(10, int(len(alerts_df) * pct))
                k = min(k, len(alerts_df))
                
                # Get ground truth
                gt_ap = compute_ground_truth_average_precision(alerts_df)
                
                # Get prediction
                pred_ap = predict_average_precision_calibrated_model(model, alerts_df, k)
                
                # Store results
                predictions.append(pred_ap)
                ground_truths.append(gt_ap)
                sample_sizes.append(k)
                errors.append(abs(pred_ap - gt_ap))
                
                # Track per-camera trajectory
                camera_id = camera_data['file_path']
                if camera_id not in camera_results:
                    camera_results[camera_id] = {'percentages': [], 'predictions': [], 'ground_truth': gt_ap}
                camera_results[camera_id]['percentages'].append(pct)
                camera_results[camera_id]['predictions'].append(pred_ap)
                
                valid_cameras += 1
                
            except Exception as e:
                print(f"⚠️  Error processing camera: {e}")
                continue
        
        if len(predictions) > 0:
            # Calculate metrics
            r2_ap = r2_score(ground_truths, predictions)
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            avg_sample_size = np.mean(sample_sizes)
            
            # Store results
            results['percentage'].append(pct)
            results['avg_sample_size'].append(avg_sample_size)
            results['r2_ap'].append(r2_ap)
            results['mean_error'].append(mean_error)
            results['std_error'].append(std_error)
            results['num_valid_cameras'].append(valid_cameras)
            
            print(f"   {pct*100:4.0f}%: R²={r2_ap:.3f}, Avg Error={mean_error:.4f}±{std_error:.4f}, "
                  f"Avg k={avg_sample_size:.0f}, Cameras={valid_cameras}")
    
    # Create comprehensive plots
    print("\n📊 Creating analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Precision-Calibrated Model Convergence Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: R² vs Sample Size Percentage
    ax1 = axes[0, 0]
    ax1.plot(np.array(results['percentage']) * 100, results['r2_ap'], 
             'bo-', linewidth=2, markersize=6, label='Average Precision R²')
    ax1.set_xlabel('Sample Size (%)')
    ax1.set_ylabel('R² Score')
    ax1.set_title('📈 R² Score vs Sample Size')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 1])
    
    # Plot 2: Error vs Sample Size
    ax2 = axes[0, 1]
    percentages_plot = np.array(results['percentage']) * 100
    ax2.errorbar(percentages_plot, results['mean_error'], yerr=results['std_error'], 
                 fmt='ro-', linewidth=2, markersize=6, capsize=5, label='Mean ± Std Error')
    ax2.set_xlabel('Sample Size (%)')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('📉 Prediction Error vs Sample Size')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Individual Camera Trajectories (sample)
    ax3 = axes[1, 0]
    # Plot a sample of camera trajectories
    sample_cameras = list(camera_results.keys())[:20]  # Plot first 20 cameras
    for camera_id in sample_cameras:
        cam_data = camera_results[camera_id]
        percentages_cam = np.array(cam_data['percentages']) * 100
        predictions_cam = cam_data['predictions']
        gt_ap = cam_data['ground_truth']
        
        # Plot trajectory
        ax3.plot(percentages_cam, predictions_cam, 'o-', alpha=0.6, linewidth=1, markersize=3)
        # Plot ground truth line
        ax3.axhline(y=gt_ap, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
    
    ax3.set_xlabel('Sample Size (%)')
    ax3.set_ylabel('Predicted Average Precision')
    ax3.set_title('🎯 Individual Camera Trajectories (Sample)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency Analysis (R² per Alert)
    ax4 = axes[1, 1]
    efficiency = np.array(results['r2_ap']) / np.array(results['avg_sample_size'])
    ax4.plot(percentages_plot, efficiency, 'go-', linewidth=2, markersize=6, label='R² per Alert')
    ax4.set_xlabel('Sample Size (%)')
    ax4.set_ylabel('R² / Avg Sample Size')
    ax4.set_title('⚡ Model Efficiency')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = "precision_calibrated_convergence_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"💾 Plot saved to {plot_path}")
    
    plt.show()
    
    # Print summary
    print(f"\n📋 PRECISION-CALIBRATED MODEL SUMMARY")
    print("=" * 45)
    best_r2_idx = np.argmax(results['r2_ap'])
    best_percentage = results['percentage'][best_r2_idx] * 100
    best_r2 = results['r2_ap'][best_r2_idx]
    best_sample_size = results['avg_sample_size'][best_r2_idx]
    
    print(f"🎯 Best R² Score: {best_r2:.4f} at {best_percentage:.0f}% ({best_sample_size:.0f} alerts avg)")
    print(f"📊 R² at 100% data: {results['r2_ap'][-1]:.4f}")
    print(f"📈 R² improvement over sample sizes: {results['r2_ap'][-1] - results['r2_ap'][0]:.4f}")
    
    # Find where R² plateaus (less than 1% improvement)
    plateau_threshold = 0.01
    for i in range(1, len(results['r2_ap'])):
        improvement = results['r2_ap'][i] - results['r2_ap'][i-1]
        if improvement < plateau_threshold:
            plateau_pct = results['percentage'][i] * 100
            plateau_r2 = results['r2_ap'][i]
            print(f"🏔️  Performance plateau: R²={plateau_r2:.4f} at {plateau_pct:.0f}%")
            break
    
    print(f"⚠️  Error at 100%: {results['mean_error'][-1]:.4f} ± {results['std_error'][-1]:.4f}")
    
    return results

if __name__ == "__main__":
    results = evaluate_convergence_precision_calibrated()
    print("✅ Precision-calibrated convergence analysis complete!") 