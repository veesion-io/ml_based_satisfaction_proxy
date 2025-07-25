#!/usr/bin/env python3
"""
Final Model Comparison: Ratio vs Refined Asymmetric
Test which model performs better on average precision accuracy
"""

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import model loaders
from load_best_model_with_ratio import (
    load_best_model_with_ratio, 
    predict_densities_and_ratio,
    calculate_precision_from_predictions
)
from load_refined_asymmetric_model import (
    load_refined_asymmetric_model, 
    predict_densities_and_ratio_refined
)

def calculate_ground_truth_average_precision(alerts_df):
    """Calculate ground truth average precision for a camera"""
    # Create probability bins
    bins = np.linspace(0, 1, 21)  # 20 bins to match model
    
    precision_values = []
    weights = []
    
    for i in range(len(bins) - 1):
        # Get alerts in this probability bin
        mask = (alerts_df['max_proba'] >= bins[i]) & (alerts_df['max_proba'] < bins[i+1])
        bin_data = alerts_df[mask]
        
        if len(bin_data) == 0:
            continue
        
        tp_count = len(bin_data[bin_data['is_theft'] == 1])
        total_count = len(bin_data)
        precision = tp_count / total_count
        
        precision_values.append(precision)
        weights.append(total_count)
    
    if not precision_values:
        return 0.0
    
    # Weighted average precision
    precision_values = np.array(precision_values)
    weights = np.array(weights)
    average_precision = np.average(precision_values, weights=weights)
    
    return average_precision

def predict_average_precision_ratio_model(model, alerts_df, k):
    """Predict average precision using ratio model"""
    tp_density, fp_density, tp_ratio = predict_densities_and_ratio(model, alerts_df, k)
    
    # Calculate precision for each bin
    precision_per_bin = calculate_precision_from_predictions(tp_density, fp_density, tp_ratio)
    
    # Calculate weighted average precision
    # Use density as weights (representing relative frequency in each bin)
    total_density = tp_density + fp_density
    weights = total_density / np.sum(total_density + 1e-9)
    avg_precision = np.average(precision_per_bin, weights=weights)
    
    return avg_precision

def predict_average_precision_refined_model(model, alerts_df, k):
    """Predict average precision using refined asymmetric model"""
    tp_density, fp_density, tp_ratio = predict_densities_and_ratio_refined(model, alerts_df, k)
    
    # Calculate precision for each bin (same as ratio model)
    fp_ratio = 1.0 - tp_ratio
    tp_scaled = tp_density * tp_ratio
    fp_scaled = fp_density * fp_ratio
    precision_per_bin = tp_scaled / (tp_scaled + fp_scaled + 1e-9)
    
    # Calculate weighted average precision
    total_density = tp_density + fp_density
    weights = total_density / np.sum(total_density + 1e-9)
    avg_precision = np.average(precision_per_bin, weights=weights)
    
    return avg_precision

def main():
    print("🔬 FINAL COMPARISON: RATIO vs REFINED ASYMMETRIC")
    print("=" * 60)
    
    # Load models
    print("Loading models...")
    ratio_model, _ = load_best_model_with_ratio()  # Unpack the tuple
    refined_model = load_refined_asymmetric_model()
    
    # Load test data
    with open("ground_truth_histograms.pkl", 'rb') as f:
        data = pickle.load(f)
    
    # Select test cameras (sufficient data for meaningful analysis)
    test_cameras = [d for d in data if len(pd.read_parquet(d['file_path'])) >= 1000][:100]
    print(f"Testing on {len(test_cameras)} cameras with ≥1000 alerts")
    
    # Test proportions
    proportions = [0.05, 0.2, 0.5, 0.8]
    
    results = []
    
    for prop in proportions:
        print(f"\n📊 Testing at {prop*100:.0f}% data proportion...")
        
        prop_results = []
        
        for i, camera_data in enumerate(tqdm(test_cameras, desc=f"Prop {prop}")):
            try:
                alerts_df = pd.read_parquet(camera_data['file_path'])
                k = max(10, int(len(alerts_df) * prop))
                
                # Ground truth
                gt_ap = calculate_ground_truth_average_precision(alerts_df)
                
                # Ratio model prediction
                ratio_ap = predict_average_precision_ratio_model(ratio_model, alerts_df, k)
                
                # Refined asymmetric model prediction
                refined_ap = predict_average_precision_refined_model(refined_model, alerts_df, k)
                
                prop_results.append({
                    'camera_idx': i,
                    'proportion': prop,
                    'gt_ap': gt_ap,
                    'ratio_ap': ratio_ap,
                    'refined_ap': refined_ap,
                    'ratio_error': abs(ratio_ap - gt_ap),
                    'refined_error': abs(refined_ap - gt_ap),
                    'improvement': abs(refined_ap - gt_ap) - abs(ratio_ap - gt_ap)  # Negative = better
                })
                
            except Exception as e:
                print(f"Error with camera {i}: {e}")
                continue
        
        # Calculate summary statistics
        if prop_results:
            avg_ratio_error = np.mean([r['ratio_error'] for r in prop_results])
            avg_refined_error = np.mean([r['refined_error'] for r in prop_results])
            avg_improvement = np.mean([r['improvement'] for r in prop_results])
            improvement_pct = (avg_improvement / avg_ratio_error) * 100 if avg_ratio_error > 0 else 0
            
            print(f"   Ratio Model Error:      {avg_ratio_error:.4f}")
            print(f"   Refined Model Error:    {avg_refined_error:.4f}")
            print(f"   Improvement:            {improvement_pct:+.1f}%")
            
            results.extend(prop_results)
    
    # Final summary
    print(f"\n📈 FINAL COMPARISON SUMMARY")
    print("=" * 40)
    
    for prop in proportions:
        prop_data = [r for r in results if r['proportion'] == prop]
        if prop_data:
            avg_improvement = np.mean([r['improvement'] for r in prop_data])
            avg_ratio_error = np.mean([r['ratio_error'] for r in prop_data])
            improvement_pct = (avg_improvement / avg_ratio_error) * 100 if avg_ratio_error > 0 else 0
            print(f"   {prop*100:2.0f}% data: {improvement_pct:+5.1f}% improvement")
    
    # Overall performance
    if results:
        overall_improvement = np.mean([r['improvement'] for r in results])
        overall_ratio_error = np.mean([r['ratio_error'] for r in results])
        overall_improvement_pct = (overall_improvement / overall_ratio_error) * 100 if overall_ratio_error > 0 else 0
        
        print(f"\n🎯 OVERALL: {overall_improvement_pct:+.1f}% improvement")
        
        if overall_improvement_pct < 0:
            print("✅ REFINED ASYMMETRIC model improved precision accuracy!")
            print("🎉 Your hypothesis was CORRECT! Biasing ratios helps!")
        else:
            print("❌ REFINED ASYMMETRIC model made things worse")
            print("🤔 Need further refinement")
    
    # Create comparison plot
    create_comparison_plot(results, proportions)
    
    return results

def create_comparison_plot(results, proportions):
    """Create comparison plot between models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Error comparison
    ratio_errors = []
    refined_errors = []
    prop_labels = []
    
    for prop in proportions:
        prop_data = [r for r in results if r['proportion'] == prop]
        if prop_data:
            ratio_errors.append(np.mean([r['ratio_error'] for r in prop_data]))
            refined_errors.append(np.mean([r['refined_error'] for r in prop_data]))
            prop_labels.append(f"{prop*100:.0f}%")
    
    x = np.arange(len(prop_labels))
    width = 0.35
    
    ax1.bar(x - width/2, ratio_errors, width, label='Ratio Model', color='steelblue', alpha=0.7)
    ax1.bar(x + width/2, refined_errors, width, label='Refined Asymmetric', color='green', alpha=0.7)
    
    ax1.set_xlabel('Data Proportion')
    ax1.set_ylabel('Average Precision Error')
    ax1.set_title('Model Comparison: Average Precision Error')
    ax1.set_xticks(x)
    ax1.set_xticklabels(prop_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Improvement percentage
    improvements = []
    for prop in proportions:
        prop_data = [r for r in results if r['proportion'] == prop]
        if prop_data:
            avg_improvement = np.mean([r['improvement'] for r in prop_data])
            avg_ratio_error = np.mean([r['ratio_error'] for r in prop_data])
            improvement_pct = (avg_improvement / avg_ratio_error) * 100 if avg_ratio_error > 0 else 0
            improvements.append(improvement_pct)
    
    colors = ['green' if imp < 0 else 'red' for imp in improvements]
    bars = ax2.bar(prop_labels, improvements, color=colors, alpha=0.7)
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Data Proportion')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Refined Asymmetric Improvement\n(Negative = Better)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.annotate(f'{imp:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10)
    
    plt.tight_layout()
    plt.savefig('final_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Final comparison plot saved to final_model_comparison.png")

if __name__ == "__main__":
    main() 