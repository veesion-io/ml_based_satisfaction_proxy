#!/usr/bin/env python3
"""
Compare ratio model vs precision-calibrated model
Focus on average precision prediction accuracy
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm

# Import both models
from load_best_model_with_ratio import (
    load_best_model_with_ratio, 
    predict_average_precision_ratio_model
)
from load_precision_calibrated_model import (
    load_precision_calibrated_model,
    predict_average_precision_calibrated_model,
    compute_ground_truth_average_precision
)

def compare_precision_models():
    """Compare ratio model vs precision-calibrated model"""
    
    print("🔍 COMPARING RATIO vs PRECISION-CALIBRATED MODELS")
    print("=" * 55)
    
    # Load both models
    print("📂 Loading models...")
    ratio_model, _ = load_best_model_with_ratio()  # Returns (model, info)
    calibrated_model = load_precision_calibrated_model()
    
    # Load test data
    print("📂 Loading camera data...")
    with open("ground_truth_histograms.pkl", 'rb') as f:
        data = pickle.load(f)
    
    # Use a subset for comparison
    num_cameras = min(100, len(data))
    test_cameras = np.random.choice(data, size=num_cameras, replace=False)
    
    print(f"📊 Comparing on {num_cameras} cameras...")
    
    # Test different sample sizes
    sample_sizes = [50, 100, 200, 500, 1000]
    
    results = {
        'sample_size': [],
        'ratio_r2': [],
        'calibrated_r2': [],
        'ratio_mean_error': [],
        'calibrated_mean_error': [],
        'ratio_std_error': [],
        'calibrated_std_error': [],
        'improvement': []
    }
    
    print("\n🧮 Evaluating across sample sizes...")
    
    for k in tqdm(sample_sizes, desc="Sample sizes"):
        ratio_predictions = []
        calibrated_predictions = []
        ground_truths = []
        ratio_errors = []
        calibrated_errors = []
        
        valid_cameras = 0
        
        for camera_data in test_cameras:
            try:
                # Load camera alerts
                alerts_df = pd.read_parquet(camera_data['file_path'])
                
                if len(alerts_df) < k:
                    continue
                
                # Limit to 2000 alerts max for model compatibility
                if len(alerts_df) > 2000:
                    alerts_df = alerts_df.sample(n=2000, replace=False)
                
                if k > len(alerts_df):
                    k_actual = len(alerts_df)
                else:
                    k_actual = k
                
                # Get ground truth
                gt_ap = compute_ground_truth_average_precision(alerts_df)
                
                # Get predictions from both models
                ratio_ap = predict_average_precision_ratio_model(ratio_model, alerts_df, k_actual)
                calibrated_ap = predict_average_precision_calibrated_model(calibrated_model, alerts_df, k_actual)
                
                # Store results
                ratio_predictions.append(ratio_ap)
                calibrated_predictions.append(calibrated_ap)
                ground_truths.append(gt_ap)
                
                ratio_errors.append(abs(ratio_ap - gt_ap))
                calibrated_errors.append(abs(calibrated_ap - gt_ap))
                
                valid_cameras += 1
                
            except Exception as e:
                print(f"⚠️  Error processing camera: {e}")
                continue
        
        if len(ratio_predictions) > 0:
            # Calculate metrics for both models
            ratio_r2 = r2_score(ground_truths, ratio_predictions)
            calibrated_r2 = r2_score(ground_truths, calibrated_predictions)
            
            ratio_mean_error = np.mean(ratio_errors)
            calibrated_mean_error = np.mean(calibrated_errors)
            
            ratio_std_error = np.std(ratio_errors)
            calibrated_std_error = np.std(calibrated_errors)
            
            improvement = calibrated_r2 - ratio_r2
            
            # Store results
            results['sample_size'].append(k)
            results['ratio_r2'].append(ratio_r2)
            results['calibrated_r2'].append(calibrated_r2)
            results['ratio_mean_error'].append(ratio_mean_error)
            results['calibrated_mean_error'].append(calibrated_mean_error)
            results['ratio_std_error'].append(ratio_std_error)
            results['calibrated_std_error'].append(calibrated_std_error)
            results['improvement'].append(improvement)
            
            print(f"   k={k:4d}: Ratio R²={ratio_r2:.3f}, Calibrated R²={calibrated_r2:.3f}, "
                  f"Improvement={improvement:+.3f}")
    
    # Create comparison plots
    print("\n📊 Creating comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Ratio Model vs Precision-Calibrated Model Comparison', fontsize=16, fontweight='bold')
    
    sample_sizes_plot = results['sample_size']
    
    # Plot 1: R² Comparison
    ax1 = axes[0, 0]
    ax1.plot(sample_sizes_plot, results['ratio_r2'], 'bo-', linewidth=2, markersize=6, label='Ratio Model')
    ax1.plot(sample_sizes_plot, results['calibrated_r2'], 'ro-', linewidth=2, markersize=6, label='Precision-Calibrated')
    ax1.set_xlabel('Sample Size')
    ax1.set_ylabel('R² Score')
    ax1.set_title('📈 R² Score Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 1])
    
    # Plot 2: Error Comparison
    ax2 = axes[0, 1]
    ax2.errorbar(sample_sizes_plot, results['ratio_mean_error'], yerr=results['ratio_std_error'], 
                 fmt='bo-', linewidth=2, markersize=6, capsize=5, label='Ratio Model')
    ax2.errorbar(sample_sizes_plot, results['calibrated_mean_error'], yerr=results['calibrated_std_error'], 
                 fmt='ro-', linewidth=2, markersize=6, capsize=5, label='Precision-Calibrated')
    ax2.set_xlabel('Sample Size')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('📉 Error Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Improvement (Calibrated - Ratio)
    ax3 = axes[1, 0]
    colors = ['green' if x > 0 else 'red' for x in results['improvement']]
    ax3.bar(sample_sizes_plot, results['improvement'], color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Sample Size')
    ax3.set_ylabel('R² Improvement (Calibrated - Ratio)')
    ax3.set_title('📊 Precision Calibration Improvement')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Direct Prediction Comparison (scatter plot)
    ax4 = axes[1, 1]
    
    # Get predictions for mid-range sample size for scatter plot
    mid_k = 200
    ratio_preds_scatter = []
    calibrated_preds_scatter = []
    gt_scatter = []
    
    for camera_data in test_cameras[:50]:  # Use subset for cleaner plot
        try:
            alerts_df = pd.read_parquet(camera_data['file_path'])
            if len(alerts_df) < mid_k:
                continue
            
            # Limit to 2000 alerts max
            if len(alerts_df) > 2000:
                alerts_df = alerts_df.sample(n=2000, replace=False)
            
            gt_ap = compute_ground_truth_average_precision(alerts_df)
            ratio_ap = predict_average_precision_ratio_model(ratio_model, alerts_df, mid_k)
            calibrated_ap = predict_average_precision_calibrated_model(calibrated_model, alerts_df, mid_k)
            
            ratio_preds_scatter.append(ratio_ap)
            calibrated_preds_scatter.append(calibrated_ap)
            gt_scatter.append(gt_ap)
            
        except:
            continue
    
    if len(ratio_preds_scatter) > 0:
        ax4.scatter(gt_scatter, ratio_preds_scatter, alpha=0.6, color='blue', label='Ratio Model')
        ax4.scatter(gt_scatter, calibrated_preds_scatter, alpha=0.6, color='red', label='Precision-Calibrated')
        
        # Perfect prediction line
        min_val = min(min(gt_scatter), min(ratio_preds_scatter + calibrated_preds_scatter))
        max_val = max(max(gt_scatter), max(ratio_preds_scatter + calibrated_preds_scatter))
        ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
        
        ax4.set_xlabel('Ground Truth Average Precision')
        ax4.set_ylabel('Predicted Average Precision')
        ax4.set_title(f'🎯 Prediction Accuracy (k={mid_k})')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = "precision_model_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"💾 Comparison plot saved to {plot_path}")
    
    plt.show()
    
    # Print summary
    print(f"\n📋 MODEL COMPARISON SUMMARY")
    print("=" * 40)
    
    avg_ratio_r2 = np.mean(results['ratio_r2'])
    avg_calibrated_r2 = np.mean(results['calibrated_r2'])
    avg_improvement = np.mean(results['improvement'])
    
    print(f"📊 Average R² - Ratio Model: {avg_ratio_r2:.4f}")
    print(f"📊 Average R² - Calibrated Model: {avg_calibrated_r2:.4f}")
    print(f"📈 Average Improvement: {avg_improvement:+.4f}")
    
    # Check if calibration helps
    improvements_positive = sum(1 for x in results['improvement'] if x > 0)
    total_tests = len(results['improvement'])
    
    print(f"✅ Calibration improves performance in {improvements_positive}/{total_tests} cases ({100*improvements_positive/total_tests:.1f}%)")
    
    if avg_improvement > 0:
        print("🎯 CONCLUSION: Precision calibration shows improvement!")
    else:
        print("⚠️  CONCLUSION: Precision calibration needs refinement")
    
    return results

if __name__ == "__main__":
    results = compare_precision_models()
    print("✅ Model comparison complete!") 