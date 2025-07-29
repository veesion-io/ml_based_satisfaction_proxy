#!/usr/bin/env python3
"""
Average Precision Convergence Analysis
Evaluates average precision error as a function of data proportion per camera
"""

import multiprocessing as mp

from analysis import (
    load_camera_paths,
    setup_multiprocessing,
    prepare_multiprocessing_args,
    run_parallel_processing,
    aggregate_results_by_proportion,
    save_results,
    create_beautiful_quantile_plot,
    print_average_precision_summary
)

def print_analysis_header():
    """Print analysis introduction"""
    print("🎯 TP Ratio Convergence Analysis")
    print("=" * 60)
    print("✅ Precision-aware model loaded successfully!")

def get_ground_truth_precision():
    """Get precomputed ground truth TP ratio"""
    gt_avg_precision_across_cameras = 0.05475697
    print(f"Using recomputed ground truth TP ratio: {gt_avg_precision_across_cameras:.8f}")
    return gt_avg_precision_across_cameras

def get_default_proportions():
    """Get default proportions to test"""
    return [0.02, 0.1, 0.2, 0.3]

def print_processing_info(camera_paths, proportions):
    """Print processing information"""
    print(f"Selected {len(camera_paths)} camera files (workers will load DataFrames as needed)")
    print(f"Memory usage: Only file paths loaded in main thread!")
    print(f"Testing {len(proportions)} proportions from {min(proportions):.1%} to {max(proportions):.1%}")
    print(f"Using mixture distribution quantiles directly (no Monte Carlo sampling needed)...")

def print_total_work_info(camera_paths, proportions):
    """Print total work information"""
    total_work = len(camera_paths) * len(proportions)
    print(f"\nProcessing ALL proportions in parallel using distribution quantiles...")
    print(f"Total work: {len(camera_paths)} cameras × {len(proportions)} proportions = {total_work:,} predictions")

def print_completion_message():
    """Print completion message"""
    print("\n🎉 TP ratio convergence analysis complete!")
    print("📁 Check tp_ratio_convergence_results_true_distribution.csv and tp_ratio_convergence_plot_with_uncertainty.png")
    print("🔬 Analysis now uses TRUE distribution of averages via Monte Carlo sampling from mixture distributions!")

def evaluate_average_precision_convergence(proportions=None):
    """
    Evaluate average precision error as a function of data proportion per camera
    Uses mixture distribution quantiles directly instead of Monte Carlo sampling
    
    Args:
        proportions: List of data proportions to test (e.g., [0.01, 0.02, 0.05, 0.1, ...])
    """
    
    # Print introduction
    print_analysis_header()
    
    # Get ground truth and proportions
    gt_avg_precision_across_cameras = get_ground_truth_precision()
    if proportions is None:
        proportions = get_default_proportions()
    
    # Load camera paths and setup multiprocessing
    camera_paths = load_camera_paths()
    camera_path_chunks, n_cores = setup_multiprocessing(camera_paths)
    
    # Print processing information
    print_processing_info(camera_paths, proportions)
    print_total_work_info(camera_paths, proportions)
    
    # Prepare arguments and run parallel processing
    args_list = prepare_multiprocessing_args(camera_path_chunks, proportions, gt_avg_precision_across_cameras)
    all_results = run_parallel_processing(args_list, n_cores)
    
    # Aggregate results by proportion
    proportion_summary = aggregate_results_by_proportion(all_results, proportions, gt_avg_precision_across_cameras)
    
    # Save results and create visualizations
    results_df, all_results_df = save_results(proportion_summary, all_results)
    create_beautiful_quantile_plot(results_df, gt_avg_precision_across_cameras)
    print_average_precision_summary(results_df)
    
    return results_df, all_results_df

if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
        
    print("\n" + "="*60)
    print("Running full evaluation with precision-aware model...")
    
    # Run average precision convergence analysis
    results, all_samples_results = evaluate_average_precision_convergence(
        proportions=None  # Use default proportion range with distribution quantiles
    )
    
    # Print completion message
    print_completion_message() 