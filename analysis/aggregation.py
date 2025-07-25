#!/usr/bin/env python3
"""
Results aggregation utilities for convergence analysis
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

def aggregate_results_by_proportion(all_results, proportions, gt_avg_precision_across_cameras):
    """Aggregate results by proportion for quantile analysis"""
    print("Aggregating results by proportion for quantile analysis...")
    proportion_summary = []
    
    for proportion in tqdm(proportions, desc="Aggregating by proportion"):
        # Get all results for this proportion
        prop_results = [r for r in all_results if r['proportion'] == proportion]
        
        if not prop_results:
            print(f"Warning: No results for proportion {proportion}")
            continue
        
        # Compute distribution of average TP ratio across cameras
        print(f"Computing distribution of average TP ratio for proportion {proportion:.1%} using VECTORIZED sampling...")
        
        # Extract mixture parameters for all cameras
        camera_mixture_params = []
        for r in prop_results:
            camera_mixture_params.append({
                'weights': r['mixture_weights'],
                'locations': r['mixture_locations'], 
                'scales': r['mixture_scales']
            })
        
        # Compute aggregated statistics
        aggregated_stats = compute_vectorized_monte_carlo_stats(
            camera_mixture_params, gt_avg_precision_across_cameras
        )
        
        # Individual camera statistics (for comparison/analysis)
        camera_stats = compute_individual_camera_stats(prop_results)
        
        # Combine all statistics
        if prop_results and len(aggregated_stats) > 0:
            proportion_summary.append({
                'proportion': proportion,
                **aggregated_stats,
                **camera_stats,
                'n_cameras': len(prop_results),
                'mean_sample_size': np.mean([r['sample_size'] for r in prop_results])
            })
    
    return proportion_summary

def compute_vectorized_monte_carlo_stats(camera_mixture_params, gt_avg_precision_across_cameras):
    """Compute statistics using vectorized Monte Carlo sampling"""
    n_monte_carlo_samples = 1000  # Number of samples for distribution of averages
    n_cameras = len(camera_mixture_params)
    n_components = len(camera_mixture_params[0]['weights'])
    
    # Stack all mixture parameters into arrays for vectorized operations
    weights = np.stack([r['weights'] for r in camera_mixture_params])  # (C, K)
    locations = np.stack([r['locations'] for r in camera_mixture_params])  # (C, K)
    scales = np.stack([r['scales'] for r in camera_mixture_params])  # (C, K)
    
    # Sample component indices for each camera and each sample
    component_indices = np.zeros((n_monte_carlo_samples, n_cameras), dtype=int)
    for c in range(n_cameras):
        component_indices[:, c] = np.random.choice(
            n_components, size=n_monte_carlo_samples, p=weights[c]
        )
    
    # Convert to one-hot to select components
    one_hot = np.eye(n_components)[component_indices]  # (S, C, K)
    
    # Select parameters using vectorized operations
    locs = np.sum(one_hot * locations[None, :, :], axis=2)  # (S, C)
    scs = np.sum(one_hot * scales[None, :, :], axis=2)  # (S, C)
    
    # Vectorized sampling from logistic: μ + s * log(u / (1 - u))
    u = np.random.uniform(0.0001, 0.9999, size=(n_monte_carlo_samples, n_cameras))
    samples = locs + scs * np.log(u / (1 - u))
    samples = np.clip(samples, 0, 1)
    
    # Average across cameras for each sample (vectorized!)
    average_samples = samples.mean(axis=1)  # (S,)
    
    return compute_distribution_statistics(average_samples, gt_avg_precision_across_cameras, n_monte_carlo_samples)

def compute_distribution_statistics(average_samples, gt_avg_precision_across_cameras, n_monte_carlo_samples):
    """Compute statistics from the distribution of averages"""
    # Calculate error for the mean of the distribution of averages
    mean_of_averages = np.mean(average_samples)
    mean_error = abs(mean_of_averages - gt_avg_precision_across_cameras)
    
    # Compute quantiles of the TRUE distribution of averages
    q05_avg = np.percentile(average_samples, 2.5)   # 2.5th percentile ≈ 5th
    q10_avg = np.percentile(average_samples, 10)
    q25_avg = np.percentile(average_samples, 25)
    median_avg = np.percentile(average_samples, 50)
    q75_avg = np.percentile(average_samples, 75)
    q90_avg = np.percentile(average_samples, 90)
    q95_avg = np.percentile(average_samples, 97.5)  # 97.5th percentile ≈ 95th
    
    std_of_averages = np.std(average_samples)
    
    return {
        # Error statistics from the distribution of averages
        'mean_avg_precision_error': mean_error,
        'std_avg_precision_error': std_of_averages,
        'median_avg_precision_error': abs(median_avg - gt_avg_precision_across_cameras),
        'q25_avg_precision_error': abs(q25_avg - gt_avg_precision_across_cameras),
        'q75_avg_precision_error': abs(q75_avg - gt_avg_precision_across_cameras),
        'q10_avg_precision_error': abs(q10_avg - gt_avg_precision_across_cameras),
        'q90_avg_precision_error': abs(q90_avg - gt_avg_precision_across_cameras),
        'q05_avg_precision_error': abs(q05_avg - gt_avg_precision_across_cameras),
        'q95_avg_precision_error': abs(q95_avg - gt_avg_precision_across_cameras),
        'min_avg_precision_error': abs(np.min(average_samples) - gt_avg_precision_across_cameras),
        'max_avg_precision_error': abs(np.max(average_samples) - gt_avg_precision_across_cameras),
        
        # TRUE distribution of averages across cameras
        'mean_avg_precision_pred': mean_of_averages,
        'std_avg_precision_pred': std_of_averages,
        'median_avg_precision_pred': median_avg,
        'q25_avg_precision_pred': q25_avg,
        'q75_avg_precision_pred': q75_avg,
        'q10_avg_precision_pred': q10_avg,
        'q90_avg_precision_pred': q90_avg,
        'q05_avg_precision_pred': q05_avg,
        'q95_avg_precision_pred': q95_avg,
        'min_avg_precision_pred': np.min(average_samples),
        'max_avg_precision_pred': np.max(average_samples),
        
        # Width of the TRUE distribution of averages
        'true_avg_ci_width': q95_avg - q05_avg,
        'true_avg_iqr_width': q75_avg - q25_avg,
        
        # Meta info
        'n_monte_carlo_samples': n_monte_carlo_samples,
    }

def compute_individual_camera_stats(prop_results):
    """Compute individual camera statistics for comparison"""
    camera_means = [r['pred_mean'] for r in prop_results]
    camera_stds = [r['pred_std'] for r in prop_results]
    camera_ci_widths = [r['pred_p95'] - r['pred_p05'] for r in prop_results]
    camera_iqr_widths = [r['pred_p75'] - r['pred_p25'] for r in prop_results]
    
    return {
        # Individual camera uncertainty statistics (for analysis)
        'mean_model_std': np.mean(camera_stds),
        'std_model_std': np.std(camera_stds),
        'mean_ci_width': np.mean(camera_ci_widths),
        'std_ci_width': np.std(camera_ci_widths),
        'mean_iqr_width': np.mean(camera_iqr_widths),
        'std_iqr_width': np.std(camera_iqr_widths),
    }

def save_results(proportion_summary, all_results):
    """Save aggregated and detailed results to CSV files"""
    results_df = pd.DataFrame(proportion_summary)
    all_results_df = pd.DataFrame(all_results)
    
    results_df.to_csv('tp_ratio_convergence_results_true_distribution.csv', index=False)
    all_results_df.to_csv('tp_ratio_convergence_all_cameras_with_mixtures.csv', index=False)
    print(f"\n✅ Results saved to tp_ratio_convergence_results_true_distribution.csv and all_cameras_with_mixtures.csv")
    
    return results_df, all_results_df 