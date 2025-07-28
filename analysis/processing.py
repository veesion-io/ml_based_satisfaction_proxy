#!/usr/bin/env python3
"""
Camera data processing utilities for convergence analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import multiprocessing as mp
from tqdm import tqdm
import random

from .model_loader import load_precision_aware_model, predict_average_precision_aware_with_uncertainty
from .noise_tuning import NoiseTuner

def load_camera_paths():
    """Load camera file paths from ground truth data"""
    print("Loading camera file paths...")
    with open('ground_truth_histograms.pkl', 'rb') as f:
        gt_data = pickle.load(f)
    
    # Filter cameras with sufficient data (just check file existence)
    print("Filtering camera file paths...")
    camera_paths = []
    for gt in tqdm(gt_data, desc="Checking cameras"):
        file_path = gt['file_path']
        if Path(file_path).exists():  # Just check file exists
            camera_paths.append(file_path)
    
    return camera_paths

def setup_multiprocessing(camera_paths):
    """Set up multiprocessing chunks for camera processing"""
    n_cores = mp.cpu_count()
    print(f"Using {n_cores} CPU cores for efficient camera-chunk processing...")
    
    # Divide camera paths into chunks for each worker
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
    
    return camera_path_chunks, n_cores

def prepare_multiprocessing_args(camera_path_chunks, proportions, gt_avg_precision_across_cameras):
    """Prepare arguments for multiprocessing"""
    args_list = [
        (camera_path_chunk, proportions, gt_avg_precision_across_cameras, worker_id)
        for worker_id, camera_path_chunk in enumerate(camera_path_chunks)
    ]
    return args_list

def process_single_camera(camera_file_path, proportions, gt_avg_precision_across_cameras, model, noise_tuner):
    """Process a single camera for all proportions"""
    results = []
    
    try:
        # Load DataFrame only when needed in worker
        camera_data = pd.read_parquet(camera_file_path)
        
        # Skip cameras with too few alerts
        if len(camera_data) < 50:
            return results
            
        camera_size = len(camera_data)
        
        # For each camera, process all proportions
        for proportion in proportions:
            # Calculate sample size for this proportion
            sample_size = max(1, int(camera_size * proportion))
            sample_size = min(sample_size, camera_size)
            
            # Sample data from camera once per proportion
            random_state = 42 + hash(camera_file_path) % 1000
            sample_data = camera_data.sample(
                n=sample_size, 
                replace=False, 
                random_state=random_state
            )
            
            # Get full uncertainty distribution from mixture model
            uncertainty_info = predict_average_precision_aware_with_uncertainty(
                model, sample_data, sample_size
            )
            
            # Adjust uncertainty with noise tuner
            adjusted_uncertainty_info = noise_tuner.tune_noise(uncertainty_info, sample_size)

            # Store result with distribution info and mixture parameters
            results.append({
                'camera_size': camera_size,
                'proportion': proportion,
                'sample_size': sample_size,
                'gt_avg_precision': gt_avg_precision_across_cameras,
                'pred_mean': adjusted_uncertainty_info['mean'],
                'pred_std': adjusted_uncertainty_info['std'],
                'pred_variance': adjusted_uncertainty_info['variance'],
                'pred_median': adjusted_uncertainty_info['median'],
                'pred_mode': adjusted_uncertainty_info['mode'],
                'pred_p05': adjusted_uncertainty_info['ci_lower'],
                'pred_p25': adjusted_uncertainty_info['p25'],
                'pred_p75': adjusted_uncertainty_info['p75'],
                'pred_p95': adjusted_uncertainty_info['ci_upper'],
                # Mixture parameters for proper averaging
                'mixture_weights': adjusted_uncertainty_info['mixture_weights'],
                'mixture_locations': adjusted_uncertainty_info['mixture_locations'],
                'mixture_scales': adjusted_uncertainty_info['mixture_scales'],
                'camera_id': camera_file_path,
            })
            
    except Exception as e:
        print(f"Error processing {camera_file_path}: {e}")
        
    return results

def process_camera_chunk(args):
    """
    Process a chunk of camera files - each worker handles multiple cameras
    """
    camera_path_chunk, proportions, gt_avg_precision_across_cameras, worker_id = args
    
    try:
        print(f"Worker {worker_id}: Loading model and processing {len(camera_path_chunk)} camera files...")
        
        # Load precision-aware model once per worker
        model = load_precision_aware_model()
        
        # Initialize noise tuner
        noise_tuner = NoiseTuner()

        # Store all results for this worker
        all_worker_results = []
        
        # Process each camera file in this chunk
        for camera_idx, camera_file_path in enumerate(camera_path_chunk):
            camera_results = process_single_camera(
                camera_file_path, proportions, gt_avg_precision_across_cameras, model, noise_tuner
            )
            all_worker_results.extend(camera_results)
        
        print(f"Worker {worker_id}: Completed! Processed {len(all_worker_results)} predictions")
        return all_worker_results
        
    except Exception as e:
        print(f"Worker {worker_id}: Error - {e}")
        return []

def run_parallel_processing(args_list, n_cores):
    """Run parallel processing of camera chunks"""
    print(f"\nProcessing all proportions in parallel using distribution quantiles...")
    
    # Process all work in parallel
    with mp.Pool(processes=n_cores) as pool:
        worker_results = pool.map(process_camera_chunk, args_list)
    
    # Flatten all results from all workers
    print(f"\nCollecting results from {len(worker_results)} workers...")
    all_results = []
    for worker_result in worker_results:
        all_results.extend(worker_result)
    
    print(f"Total predictions collected: {len(all_results):,}")
    return all_results 