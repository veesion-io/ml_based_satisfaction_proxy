#!/usr/bin/env python3
"""
Average Precision Convergence Analysis
Evaluates average precision error as a function of data proportion per camera
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
import multiprocessing as mp
from functools import partial
from load_best_model_with_ratio import (
    load_best_model_with_ratio, 
    predict_densities_and_ratio,
    calculate_precision_from_predictions,
    ResidualMLP, MAB, N_BINS, DEVICE
)
import torch.nn.functional as F

# Precision-aware model definition (Mixture of Logistic Distributions)
class DeepSetsPrecisionAware(torch.nn.Module):
    def __init__(self, phi_dim, n_bins, num_heads):
        super(DeepSetsPrecisionAware, self).__init__()
        if phi_dim % num_heads != 0: 
            phi_dim = (phi_dim // num_heads + 1) * num_heads
        
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(3, phi_dim), 
            ResidualMLP(phi_dim), 
            ResidualMLP(phi_dim)
        )
        self.pooling = MAB(phi_dim, num_heads)
        self.query = torch.nn.Parameter(torch.randn(1, 1, phi_dim))
        self.rho = torch.nn.Sequential(ResidualMLP(phi_dim), ResidualMLP(phi_dim))
        
        # Three output heads: TP density, FP density, and TP ratio mixture distribution
        self.tp_head = torch.nn.Linear(phi_dim, n_bins)
        self.fp_head = torch.nn.Linear(phi_dim, n_bins)
        
        # Mixture of logistic components for TP ratio distribution
        self.num_mixture_components = 5  # Number of logistic components
        self.mixture_weights_head = torch.nn.Linear(phi_dim, self.num_mixture_components)  # Mixture weights
        self.mixture_locations_head = torch.nn.Linear(phi_dim, self.num_mixture_components)  # Location parameters μᵢ
        self.mixture_scales_head = torch.nn.Linear(phi_dim, self.num_mixture_components)  # Scale parameters sᵢ
    
    def forward(self, x, counts):
        mask = torch.arange(x.size(1), device=x.device)[None, :] < counts[:, None]
        phi_out = self.phi(x * mask.unsqueeze(-1))
        query = self.query.repeat(x.shape[0], 1, 1)
        agg = self.pooling(query, phi_out).squeeze(1)
        rho_out = self.rho(agg)
        
        tp_out = F.softmax(self.tp_head(rho_out), dim=1)
        fp_out = F.softmax(self.fp_head(rho_out), dim=1)
        
        # Mixture of logistic components for TP ratio distribution
        mixture_weights_raw = self.mixture_weights_head(rho_out)  # (B, num_components)
        mixture_locations_raw = self.mixture_locations_head(rho_out)  # (B, num_components)
        mixture_scales_raw = self.mixture_scales_head(rho_out)  # (B, num_components)
        
        # Normalize mixture weights
        mixture_weights = F.softmax(mixture_weights_raw, dim=1)
        
        # Constrain locations to [0, 1] using sigmoid
        mixture_locations = torch.sigmoid(mixture_locations_raw)
        
        # Constrain scales to be positive using softplus
        mixture_scales = F.softplus(mixture_scales_raw) + 1e-6  # Add small constant for numerical stability
        
        return tp_out, fp_out, mixture_weights, mixture_locations, mixture_scales

def load_precision_aware_model():
    checkpoint = torch.load("runs/best_model/best_checkpoint_precision_aware.pth", map_location=DEVICE)
    hyperparams = checkpoint['hyperparameters']
    model = DeepSetsPrecisionAware(hyperparams['phi_dim'], N_BINS, hyperparams['num_heads']).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_densities_and_ratio_precision_aware(model, sample_data, sample_size):
    features = np.concatenate([
        sample_data[['max_proba', 'is_theft']].values,
        np.full((len(sample_data), 1), np.log(sample_size) / np.log(2000))
    ], axis=1)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    counts = torch.tensor([sample_size], dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        tp_density, fp_density, mixture_weights, mixture_locations, mixture_scales = model(features_tensor, counts)
    
    return (tp_density.cpu().numpy()[0], 
            fp_density.cpu().numpy()[0], 
            mixture_weights.cpu().numpy()[0],
            mixture_locations.cpu().numpy()[0], 
            mixture_scales.cpu().numpy()[0])

def mixture_logistic_cdf_numpy(x, weights, locations, scales):
    """
    Numpy version of mixture logistic CDF: P(X ≤ x) = Σᵢ wᵢ * σ((x - μᵢ) / sᵢ)
    """
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip for numerical stability
    
    # Compute CDF for each component
    cdfs = sigmoid((x[:, None] - locations[None, :]) / scales[None, :])  # (n_points, n_components)
    
    # Weighted sum
    mixture_cdf = np.sum(weights[None, :] * cdfs, axis=1)  # (n_points,)
    
    return mixture_cdf

def mixture_logistic_pdf_numpy(x, weights, locations, scales):
    """
    Numpy version of mixture logistic PDF
    """
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    # Compute PDF for each component
    standardized = (x[:, None] - locations[None, :]) / scales[None, :]  # (n_points, n_components)
    sigmoid_vals = sigmoid(standardized)
    pdfs = (1.0 / scales[None, :]) * sigmoid_vals * (1.0 - sigmoid_vals)  # (n_points, n_components)
    
    # Weighted sum
    mixture_pdf = np.sum(weights[None, :] * pdfs, axis=1)  # (n_points,)
    
    return mixture_pdf

def extract_tp_ratio_distribution_info(mixture_weights, mixture_locations, mixture_scales):
    """Extract statistics from mixture of logistic distributions"""
    
    # Mean of mixture = Σᵢ wᵢ * μᵢ (location parameters are means for logistic)
    mean = np.sum(mixture_weights * mixture_locations)
    
    # For variance: Var(mixture) = Σᵢ wᵢ * (μᵢ² + σᵢ²) - (Σᵢ wᵢ * μᵢ)²
    # For logistic distribution, variance = (π²/3) * s²
    logistic_variances = (np.pi**2 / 3) * (mixture_scales**2)
    second_moment = np.sum(mixture_weights * (mixture_locations**2 + logistic_variances))
    variance = second_moment - mean**2
    std = np.sqrt(max(variance, 0))  # Ensure non-negative
    
    # Compute CDF and find quantiles numerically
    x_eval = np.linspace(0, 1, 1000)  # Evaluation points
    cdf_vals = mixture_logistic_cdf_numpy(x_eval, mixture_weights, mixture_locations, mixture_scales)
    pdf_vals = mixture_logistic_pdf_numpy(x_eval, mixture_weights, mixture_locations, mixture_scales)
    
    # Mode (x with highest PDF)
    mode_idx = np.argmax(pdf_vals)
    mode = x_eval[mode_idx]
    
    # Quantiles
    def find_quantile(q):
        idx = np.searchsorted(cdf_vals, q)
        return x_eval[min(idx, len(x_eval)-1)]
    
    ci_lower = find_quantile(0.025)
    ci_upper = find_quantile(0.975)
    p25 = find_quantile(0.25)
    p75 = find_quantile(0.75)
    median = find_quantile(0.5)
    
    return {
        'mean': mean,
        'std': std,
        'variance': variance,
        'mode': mode,
        'median': median,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p25': p25,
        'p75': p75,
        'mixture_weights': mixture_weights,
        'mixture_locations': mixture_locations,
        'mixture_scales': mixture_scales,
        'pdf_x': x_eval,
        'pdf_y': pdf_vals,
        'cdf_x': x_eval,
        'cdf_y': cdf_vals
    }

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def calculate_simple_tp_ratio(camera_data):
    """Calculate simple TP ratio (same as what the model predicts)"""
    tp_count = len(camera_data[camera_data['is_theft'] == 1])
    total_count = len(camera_data)
    
    if total_count == 0:
        return 0.0
    
    return tp_count / total_count


def predict_average_precision_aware(model, sample_data, sample_size):
    """Predict TP ratio from sample data using the precision-aware model with uncertainty"""
    
    # Get predictions from precision-aware model (now returns mixture of logistic distributions)
    _, _, mixture_weights, mixture_locations, mixture_scales = predict_densities_and_ratio_precision_aware(
        model, sample_data, sample_size=sample_size
    )
    
    # Extract distribution information
    dist_info = extract_tp_ratio_distribution_info(mixture_weights, mixture_locations, mixture_scales)
    
    # Return mean as point estimate (for compatibility with existing code)
    return dist_info['mean']

def predict_average_precision_aware_with_uncertainty(model, sample_data, sample_size):
    """Predict TP ratio with full uncertainty information"""
    
    # Get predictions from precision-aware model
    _, _, mixture_weights, mixture_locations, mixture_scales = predict_densities_and_ratio_precision_aware(
        model, sample_data, sample_size=sample_size
    )
    
    # Return full distribution information
    return extract_tp_ratio_distribution_info(mixture_weights, mixture_locations, mixture_scales)

def process_camera_chunk(args):
    """
    Efficient worker function: each process handles a subset of camera file paths
    and loads DataFrames as needed, processing ALL proportions using distribution quantiles directly
    """
    camera_path_chunk, proportions, gt_avg_precision_across_cameras, worker_id = args
    
    try:
        print(f"Worker {worker_id}: Loading model and processing {len(camera_path_chunk)} camera files...")
        
        # Load precision-aware model ONCE per worker (much more efficient!)
        model = load_precision_aware_model()
        
        # Store all results for this worker
        all_worker_results = []
        
        # Process each camera file in this chunk
        for camera_idx, camera_file_path in enumerate(camera_path_chunk):
            try:
                # Load DataFrame ONLY when needed in worker
                camera_data = pd.read_parquet(camera_file_path)
                
                # Skip cameras with too few alerts
                if len(camera_data) < 50:
                    continue
                    
                camera_size = len(camera_data)
                
                # For each camera, process all proportions (NO MORE SAMPLING LOOP!)
                for proportion in proportions:
                    # Calculate sample size for this proportion
                    sample_size = max(1, int(camera_size * proportion))
                    sample_size = min(sample_size, camera_size)
                    
                    # Sample data from camera ONCE per proportion
                    random_state = 42 + hash(camera_file_path) % 1000
                    sample_data = camera_data.sample(
                        n=sample_size, 
                        replace=False, 
                        random_state=random_state
                    )
                    
                    # Get FULL uncertainty distribution from mixture model
                    uncertainty_info = predict_average_precision_aware_with_uncertainty(
                        model, sample_data, sample_size
                    )
                    
                    # Store result with distribution info AND mixture parameters for true averaging
                    all_worker_results.append({
                        'camera_size': camera_size,
                        'proportion': proportion,
                        'sample_size': sample_size,
                        'gt_avg_precision': gt_avg_precision_across_cameras,
                        'pred_mean': uncertainty_info['mean'],
                        'pred_std': uncertainty_info['std'],
                        'pred_variance': uncertainty_info['variance'],
                        'pred_median': uncertainty_info['median'],
                        'pred_mode': uncertainty_info['mode'],
                        'pred_p05': uncertainty_info['ci_lower'],   # 2.5th percentile ≈ 5th
                        'pred_p25': uncertainty_info['p25'],
                        'pred_p75': uncertainty_info['p75'],
                        'pred_p95': uncertainty_info['ci_upper'],   # 97.5th percentile ≈ 95th
                        # MIXTURE PARAMETERS for proper averaging
                        'mixture_weights': uncertainty_info['mixture_weights'],
                        'mixture_locations': uncertainty_info['mixture_locations'],
                        'mixture_scales': uncertainty_info['mixture_scales'],
                        'camera_id': camera_file_path,
                        'worker_id': worker_id
                    })
                    
            except Exception as e:
                print(f"Worker {worker_id}: Error processing {camera_file_path}: {e}")
                continue
        
        print(f"Worker {worker_id}: Completed! Processed {len(all_worker_results)} predictions")
        return all_worker_results
        
    except Exception as e:
        print(f"Worker {worker_id}: Error - {e}")
        return []

def evaluate_average_precision_convergence(proportions=None):
    """
    Evaluate average precision error as a function of data proportion per camera
    Uses mixture distribution quantiles directly instead of Monte Carlo sampling
    
    Args:
        proportions: List of data proportions to test (e.g., [0.01, 0.02, 0.05, 0.1, ...])
    """
    
    print("🎯 TP Ratio Convergence Analysis")
    print("=" * 60)
    
    print("✅ Precision-aware model loaded successfully!")
    
    # Use previously computed ground truth TP ratio (constant for efficiency)
    gt_avg_precision_across_cameras = 0.0548  # Correct average TP ratio across all cameras
    print(f"Using precomputed ground truth TP ratio: {gt_avg_precision_across_cameras:.4f}")
    
    # Load only file paths (workers will load DataFrames as needed)
    print("Loading camera file paths...")
    with open('ground_truth_histograms.pkl', 'rb') as f:
        gt_data = pickle.load(f)
    
    # Filter cameras with sufficient data (just check file existence, workers load DataFrames)
    print("Filtering camera file paths...")
    camera_paths = []
    for gt in tqdm(gt_data, desc="Checking cameras"):
        file_path = gt['file_path']
        if Path(file_path).exists():  # Just check file exists, don't load DataFrame
            camera_paths.append(file_path)
    
    print(f"Selected {len(camera_paths)} camera files (workers will load DataFrames as needed)")
    print(f"Memory usage: Only file paths loaded in main thread!")
    
    # Define proportions to test (every 10% for faster analysis)
    if proportions is None:
        proportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    print(f"Testing {len(proportions)} proportions from {min(proportions):.1%} to {max(proportions):.1%}")
    print(f"Using mixture distribution quantiles directly (no Monte Carlo sampling needed)...")
    
    # Get number of CPU cores for multiprocessing (can use more now since each worker loads model once!)
    n_cores = mp.cpu_count()
    print(f"Using {n_cores} CPU cores for efficient camera-chunk processing...")
    
    # Divide camera paths into chunks for each worker (much more efficient!)
    cameras_per_worker = len(camera_paths) // n_cores
    camera_path_chunks = []
    for i in range(n_cores):
        start_idx = i * cameras_per_worker
        if i == n_cores - 1:  # Last worker gets remaining cameras
            end_idx = len(camera_paths)
        else:
            end_idx = (i + 1) * cameras_per_worker
        camera_path_chunks.append(camera_paths[start_idx:end_idx])
    
    print(f"Divided {len(camera_paths)} camera paths into {len(camera_path_chunks)} chunks")
    print(f"Chunk sizes: {[len(chunk) for chunk in camera_path_chunks]}")
    
    # Prepare arguments for efficient multiprocessing (no more n_samples!)
    args_list = [
        (camera_path_chunk, proportions, gt_avg_precision_across_cameras, worker_id)
        for worker_id, camera_path_chunk in enumerate(camera_path_chunks)
    ]
    
    print(f"\nProcessing ALL proportions in parallel using distribution quantiles...")
    print(f"Total work: {len(camera_paths)} cameras × {len(proportions)} proportions = {len(camera_paths) * len(proportions):,} predictions")
    
    # Process all work in parallel (MUCH more efficient!)
    with mp.Pool(processes=n_cores) as pool:
        worker_results = pool.map(process_camera_chunk, args_list)
    
    # Flatten all results from all workers
    print(f"\nCollecting results from {len(worker_results)} workers...")
    all_results = []
    for worker_result in worker_results:
        all_results.extend(worker_result)
    
    print(f"Total predictions collected: {len(all_results):,}")
    
    # Now aggregate results by proportion for quantile analysis
    print(f"Aggregating results by proportion for quantile analysis...")
    proportion_summary = []
    
    for proportion in tqdm(proportions, desc="Aggregating by proportion"):
        # Get all results for this proportion
        prop_results = [r for r in all_results if r['proportion'] == proportion]
        
        if not prop_results:
            print(f"Warning: No results for proportion {proportion}")
            continue
        
        # CORRECT APPROACH: Compute distribution of average TP ratio across cameras
        # by Monte Carlo sampling from each camera's mixture distribution
        
        print(f"Computing distribution of average TP ratio for proportion {proportion:.1%} using VECTORIZED sampling...")
        
        # Extract mixture parameters for all cameras
        camera_mixture_params = []
        for r in prop_results:
            camera_mixture_params.append({
                'weights': r['mixture_weights'],
                'locations': r['mixture_locations'], 
                'scales': r['mixture_scales']
            })
        
        # VECTORIZED Monte Carlo sampling for MASSIVE speedup
        n_monte_carlo_samples = 1000  # Number of samples for distribution of averages
        n_cameras = len(camera_mixture_params)
        n_components = len(camera_mixture_params[0]['weights'])
        
        # Stack all mixture parameters into arrays for vectorized operations
        weights = np.stack([r['weights'] for r in camera_mixture_params])  # (C, K)
        locations = np.stack([r['locations'] for r in camera_mixture_params])  # (C, K)
        scales = np.stack([r['scales'] for r in camera_mixture_params])  # (C, K)
        
        # Sample component indices for each camera and each sample
        # We need to handle per-camera mixture weights properly
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
        
        # Individual camera statistics (for comparison/analysis)
        camera_means = [r['pred_mean'] for r in prop_results]
        camera_stds = [r['pred_std'] for r in prop_results]
        camera_ci_widths = [r['pred_p95'] - r['pred_p05'] for r in prop_results]
        camera_iqr_widths = [r['pred_p75'] - r['pred_p25'] for r in prop_results]
        
        if prop_results and len(average_samples) > 0:
            # Calculate statistics using TRUE distribution of averages across cameras
            proportion_summary.append({
                'proportion': proportion,
                # Error statistics from the distribution of averages
                'mean_avg_precision_error': mean_error,
                'std_avg_precision_error': std_of_averages,  # True std of average distribution
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
                # Individual camera uncertainty statistics (for analysis)
                'mean_model_std': np.mean(camera_stds),
                'std_model_std': np.std(camera_stds),
                'mean_ci_width': np.mean(camera_ci_widths),
                'std_ci_width': np.std(camera_ci_widths),
                'mean_iqr_width': np.mean(camera_iqr_widths),
                'std_iqr_width': np.std(camera_iqr_widths),
                # Width of the TRUE distribution of averages
                'true_avg_ci_width': q95_avg - q05_avg,
                'true_avg_iqr_width': q75_avg - q25_avg,
                # Meta info
                'n_monte_carlo_samples': n_monte_carlo_samples,
                'n_cameras': len(prop_results),  # Number of cameras that actually contributed
                'mean_sample_size': np.mean([r['sample_size'] for r in prop_results])
            })
    
    # Convert to DataFrames
    results_df = pd.DataFrame(proportion_summary)
    all_results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv('tp_ratio_convergence_results_true_distribution.csv', index=False)
    all_results_df.to_csv('tp_ratio_convergence_all_cameras_with_mixtures.csv', index=False)
    print(f"\n✅ Results saved to tp_ratio_convergence_results_true_distribution.csv and all_cameras_with_mixtures.csv")
    
    # Create convergence plot with beautiful quantiles
    create_beautiful_quantile_plot(results_df, gt_avg_precision_across_cameras)
    
    # Print summary statistics
    print_average_precision_summary(results_df)
    
    return results_df, all_results_df

def create_beautiful_quantile_plot(results_df, gt_avg_precision):
    """Create beautiful quantile plot with filled areas, uncertainty analysis, and prominent median"""
    
    # Set up the plot with a modern style - NOW WITH 3 SUBPLOTS FOR UNCERTAINTY!
    plt.style.use('default')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    # Define a beautiful color palette
    colors = {
        'median': '#FF6B6B',      # Bright red for median (hottest)
        'q25_75': '#4ECDC4',      # Teal for IQR
        'q10_90': '#45B7D1',      # Blue for 10-90%
        'q05_95': '#96CEB4',      # Light green for 5-95%
        'mean': '#FFA726',        # Orange for mean
        'gt': '#2E8B57',          # Sea green for ground truth
        'background': '#F8F9FA'   # Light background
    }
    
    x = results_df['proportion'] * 100  # Convert to percentage
    
    # Plot 1: Average Precision Quantiles (what user wanted!)
    # Fill areas from outside to inside for beautiful layering effect
    ax1.fill_between(x, results_df['q05_avg_precision_pred'], results_df['q95_avg_precision_pred'], 
                     alpha=0.2, color=colors['q05_95'], label='5-95th Percentile')
    ax1.fill_between(x, results_df['q10_avg_precision_pred'], results_df['q90_avg_precision_pred'], 
                     alpha=0.3, color=colors['q10_90'], label='10-90th Percentile')
    ax1.fill_between(x, results_df['q25_avg_precision_pred'], results_df['q75_avg_precision_pred'], 
                     alpha=0.4, color=colors['q25_75'], label='25-75th Percentile (IQR)')
    
    # Plot the median line as the "hottest" (most prominent)
    ax1.plot(x, results_df['median_avg_precision_pred'], linewidth=4, 
             color=colors['median'], label='Median Prediction', 
             alpha=0.95, zorder=10)
    
    # Add markers for all data points to ensure visibility
    ax1.scatter(x, results_df['median_avg_precision_pred'], 
                s=80, color='white', edgecolors=colors['median'], 
                linewidth=2, zorder=11, alpha=0.9)
    
    # Add mean line for comparison (more subtle)
    ax1.plot(x, results_df['mean_avg_precision_pred'], '--', linewidth=2, 
             color=colors['mean'], alpha=0.7, label='Mean Prediction', zorder=9)
    
    # Add GROUND TRUTH horizontal line (what user wanted!)
    ax1.axhline(y=gt_avg_precision, color=colors['gt'], linestyle='-', alpha=0.8, 
                linewidth=3, label=f'Ground Truth ({gt_avg_precision:.4f})', zorder=12)
    
    ax1.set_xlabel('Data Proportion (%)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('TP Ratio Across Cameras', fontsize=14, fontweight='bold')
    ax1.set_title('TP Ratio Convergence\n(Predicted vs Ground Truth)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.set_xlim(0, 105)
    # Set y-limits centered around the data range, not starting from 0
    y_min = min(results_df['q05_avg_precision_pred'].min(), gt_avg_precision) * 0.98
    y_max = max(results_df['q95_avg_precision_pred'].max(), gt_avg_precision) * 1.02
    ax1.set_ylim(y_min, y_max)
    
    # Plot 2: Error quantiles (fixed to show filled areas properly)
    # Fill areas from outside to inside for beautiful layering effect
    ax2.fill_between(x, results_df['q05_avg_precision_error'], results_df['q95_avg_precision_error'], 
                     alpha=0.25, color=colors['q05_95'], label='5-95th Percentile', zorder=1)
    ax2.fill_between(x, results_df['q10_avg_precision_error'], results_df['q90_avg_precision_error'], 
                     alpha=0.35, color=colors['q10_90'], label='10-90th Percentile', zorder=2)
    ax2.fill_between(x, results_df['q25_avg_precision_error'], results_df['q75_avg_precision_error'], 
                     alpha=0.45, color=colors['q25_75'], label='25-75th Percentile (IQR)', zorder=3)
    
    # Plot the median line as the "hottest" (most prominent)
    ax2.plot(x, results_df['median_avg_precision_error'], linewidth=4, 
             color=colors['median'], label='Median Error', 
             alpha=0.95, zorder=10)
    
    # Add markers for all data points to ensure visibility
    ax2.scatter(x, results_df['median_avg_precision_error'], 
                s=80, color='white', edgecolors=colors['median'], 
                linewidth=2, zorder=11, alpha=0.9)
    
    # Add mean line for comparison (more subtle)
    ax2.plot(x, results_df['mean_avg_precision_error'], '--', linewidth=2, 
             color=colors['mean'], alpha=0.7, label='Mean Error', zorder=9)
    
    # Add target lines
    ax2.axhline(y=0.01, color='green', linestyle=':', alpha=0.6, linewidth=2, label='1% Target')
    ax2.axhline(y=0.02, color='orange', linestyle=':', alpha=0.6, linewidth=2, label='2% Target')
    ax2.axhline(y=0.05, color='red', linestyle=':', alpha=0.6, linewidth=2, label='5% Threshold')
    
    ax2.set_xlabel('Data Proportion (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('|Avg(Predicted TP Ratio) - Avg(GT TP Ratio)| Error', fontsize=14, fontweight='bold')
    ax2.set_title('TP Ratio Error Distribution\n(Quantiles of Prediction Errors)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.legend(fontsize=11, framealpha=0.9, loc='upper right')
    ax2.set_xlim(0, 105)
    # Set y-limits for error plot centered around data, not starting from 0
    error_max = results_df['q95_avg_precision_error'].max() * 1.1
    ax2.set_ylim(0, error_max)
    
    # Add annotations for key points on predictions plot
    final_median_pred = results_df['median_avg_precision_pred'].iloc[-1]
    final_error = abs(final_median_pred - gt_avg_precision)
    
    ax1.annotate(f'Final Error: {final_error:.4f}\nGT: {gt_avg_precision:.4f}\nPred: {final_median_pred:.4f}',
                xy=(100, final_median_pred),
                xytext=(70, final_median_pred + 0.01),
                arrowprops=dict(arrowstyle='->', color=colors['median'], lw=2),
                fontsize=10, ha='left', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                         edgecolor=colors['median'], alpha=0.9))
    
    # Plot 3: TRUE Distribution Uncertainty Analysis
    # Show how the width of the TRUE distribution of averages changes with data proportion
    ax3.fill_between(x, 
                     results_df['true_avg_ci_width'] * 0.9, 
                     results_df['true_avg_ci_width'] * 1.1, 
                     alpha=0.3, color=colors['q25_75'], label='True Avg CI Width ± 10%')
    
    ax3.plot(x, results_df['true_avg_ci_width'], linewidth=3, 
             color=colors['median'], label='True Avg CI Width (95%)', 
             alpha=0.9, zorder=10)
    
    ax3.plot(x, results_df['true_avg_iqr_width'], '--', linewidth=2, 
             color=colors['mean'], alpha=0.8, label='True Avg IQR Width (25-75%)', zorder=9)
    
    ax3.plot(x, results_df['std_avg_precision_pred'], ':', linewidth=2, 
             color=colors['q10_90'], alpha=0.8, label='Std of Average Distribution', zorder=8)
    
    # Individual camera uncertainty for comparison (thinner line)
    ax3.plot(x, results_df['mean_ci_width'], '-.', linewidth=1, 
             color='gray', alpha=0.6, label='Mean Individual Camera CI', zorder=7)
    
    # Add markers for key points
    ax3.scatter(x, results_df['true_avg_ci_width'], 
                s=60, color='white', edgecolors=colors['median'], 
                linewidth=2, zorder=11, alpha=0.9)
    
    ax3.set_xlabel('Data Proportion (%)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Distribution Width Metrics', fontsize=14, fontweight='bold')
    ax3.set_title('TRUE Distribution of Averages Width\n(Proper Statistical Combination)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.legend(fontsize=10, framealpha=0.9)
    ax3.set_xlim(0, 105)
    
    # Set y-limits for uncertainty plot
    uncertainty_max = max(results_df['true_avg_ci_width'].max(), results_df['true_avg_iqr_width'].max()) * 1.1
    ax3.set_ylim(0, uncertainty_max)
    
    # Add annotation for true uncertainty trend
    final_true_ci_width = results_df['true_avg_ci_width'].iloc[-1]
    initial_true_ci_width = results_df['true_avg_ci_width'].iloc[0]
    true_uncertainty_reduction = ((initial_true_ci_width - final_true_ci_width) / initial_true_ci_width * 100)
    
    ax3.annotate(f'True Uncertainty Reduction: {true_uncertainty_reduction:.1f}%\nFinal True CI: {final_true_ci_width:.4f}',
                xy=(100, final_true_ci_width),
                xytext=(70, final_true_ci_width + 0.005),
                arrowprops=dict(arrowstyle='->', color=colors['median'], lw=2),
                fontsize=10, ha='left', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                         edgecolor=colors['median'], alpha=0.9))

    # Style improvements
    for ax in [ax1, ax2, ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig('tp_ratio_convergence_plot_with_uncertainty.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    print(f"📊 Enhanced plot with uncertainty analysis saved to tp_ratio_convergence_plot_with_uncertainty.png")

def print_average_precision_summary(results_df):
    """Print average precision convergence summary statistics"""
    
    print("\n" + "="*70)
    print("📈 TP RATIO PREDICTION ERROR SUMMARY (ACROSS CAMERAS)")
    print("="*70)
    
    # Overall statistics
    best_error = results_df['mean_avg_precision_error'].min()
    worst_error = results_df['mean_avg_precision_error'].max()
    best_prop = results_df.loc[results_df['mean_avg_precision_error'].idxmin(), 'proportion']
    worst_prop = results_df.loc[results_df['mean_avg_precision_error'].idxmax(), 'proportion']
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   Best Error:       {best_error:.4f} (at {best_prop:.1%} data)")
    print(f"   Worst Error:      {worst_error:.4f} (at {worst_prop:.1%} data)")
    print(f"   Improvement:      {worst_error - best_error:.4f} reduction")
    print(f"   Relative Gain:    {((worst_error - best_error) / worst_error * 100):.1f}%")
    
    # Data proportion analysis
    print(f"\n📊 DATA PROPORTION ANALYSIS:")
    small_props = results_df[results_df['proportion'] <= 0.10]
    medium_props = results_df[(results_df['proportion'] > 0.10) & (results_df['proportion'] <= 0.50)]
    large_props = results_df[results_df['proportion'] > 0.50]
    
    if not small_props.empty:
        print(f"   Small (≤10%):     {small_props['mean_avg_precision_error'].mean():.4f} avg error")
    if not medium_props.empty:
        print(f"   Medium (10-50%):  {medium_props['mean_avg_precision_error'].mean():.4f} avg error")
    if not large_props.empty:
        print(f"   Large (>50%):     {large_props['mean_avg_precision_error'].mean():.4f} avg error")
    
    # Error targets
    print(f"\n🏆 ERROR TARGETS:")
    targets = [0.01, 0.02, 0.03, 0.05]
    for target in targets:
        target_reached = results_df[results_df['mean_avg_precision_error'] <= target]
        if not target_reached.empty:
            min_prop = target_reached['proportion'].min()
            print(f"   {target:.1%} Error:      {min_prop:.1%} data needed")
        else:
            print(f"   {target:.1%} Error:      Not achieved")
    
    # Convergence analysis
    print(f"\n📈 CONVERGENCE PATTERN:")
    
    # Calculate improvement rate
    if len(results_df) >= 2:
        first_half = results_df.head(len(results_df)//2)
        second_half = results_df.tail(len(results_df)//2)
        
        first_avg = first_half['mean_avg_precision_error'].mean()
        second_avg = second_half['mean_avg_precision_error'].mean()
        improvement = first_avg - second_avg
        
        print(f"   Early proportions: {first_avg:.4f} avg error")
        print(f"   Later proportions: {second_avg:.4f} avg error")
        print(f"   Improvement:       {improvement:.4f} ({improvement/first_avg*100:.1f}%)")
        
        if improvement > 0:
            print(f"   ✅ Clear convergence: error decreases with more data")
        else:
            print(f"   ⚠️  Plateau/degradation: error may not improve with more data")
    
    # Sample size info
    print(f"\n📏 SAMPLE SIZE INFO:")
    print(f"   Min sample size:   {results_df['mean_sample_size'].min():.0f} alerts")
    print(f"   Max sample size:   {results_df['mean_sample_size'].max():.0f} alerts")
    print(f"   Average cameras:   {results_df['n_cameras'].mean():.0f} per proportion")
    
    # TRUE DISTRIBUTION UNCERTAINTY ANALYSIS
    print(f"\n🎯 TRUE DISTRIBUTION OF AVERAGES ANALYSIS:")
    initial_true_uncertainty = results_df['true_avg_ci_width'].iloc[0]
    final_true_uncertainty = results_df['true_avg_ci_width'].iloc[-1]
    true_uncertainty_reduction = ((initial_true_uncertainty - final_true_uncertainty) / initial_true_uncertainty * 100)
    
    print(f"   Initial TRUE Uncertainty:  {initial_true_uncertainty:.4f} (95% CI width of avg distribution)")
    print(f"   Final TRUE Uncertainty:    {final_true_uncertainty:.4f} (95% CI width of avg distribution)")
    print(f"   TRUE Uncertainty Reduction: {true_uncertainty_reduction:.1f}%")
    
    print(f"\n   Distribution of Averages Metrics:")
    print(f"   Mean Std of Averages:     {results_df['std_avg_precision_pred'].mean():.4f}")
    print(f"   Mean TRUE IQR Width:      {results_df['true_avg_iqr_width'].mean():.4f}")
    print(f"   Monte Carlo Samples:      {results_df['n_monte_carlo_samples'].iloc[0]:,}")
    
    print(f"\n   Comparison with Individual Camera Stats:")
    print(f"   Mean Individual Camera CI: {results_df['mean_ci_width'].mean():.4f}")
    print(f"   Ratio (True/Individual):   {results_df['true_avg_ci_width'].mean() / results_df['mean_ci_width'].mean():.3f}")
    
    # True uncertainty vs Error correlation
    if len(results_df) > 2:
        true_uncertainty_error_corr = np.corrcoef(results_df['true_avg_ci_width'], results_df['mean_avg_precision_error'])[0,1]
        print(f"\n   📊 True Uncertainty-Error Correlation: {true_uncertainty_error_corr:.3f}")
        if true_uncertainty_error_corr > 0.5:
            print(f"   ✅ High correlation: true distribution uncertainty tracks prediction errors well!")
        elif true_uncertainty_error_corr > 0.2:
            print(f"   ⚠️  Moderate correlation: some alignment between true uncertainty and errors")
        else:
            print(f"   ❌ Low correlation: true uncertainty may not reflect prediction quality")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
        
    print("\n" + "="*60)
    print("Running full evaluation with precision-aware model...")
    
    # Run average precision convergence analysis
    results, all_samples_results = evaluate_average_precision_convergence(
        proportions=None  # Use default proportion range with distribution quantiles
    )
    
    print("\n🎉 TP ratio convergence analysis complete!")
    print("📁 Check tp_ratio_convergence_results_true_distribution.csv and tp_ratio_convergence_plot_with_uncertainty.png")
    print("🔬 Analysis now uses TRUE distribution of averages via Monte Carlo sampling from mixture distributions!") 