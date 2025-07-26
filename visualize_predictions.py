#!/usr/bin/env python3
"""
Visualize model predictions for camera precision distributions against ground truth.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from analysis.model_loader import load_precision_aware_model, predict_densities_and_ratio_precision_aware
from models.training import load_data
from analysis.distribution_utils import mixture_logistic_pdf_numpy

def visualize_predictions(num_examples=5, data_fraction=1.0):
    """
    Generates and saves a plot comparing the model's predicted precision distributions
    with the ground truth for a random sample of cameras from the validation set.
    """
    print("Loading model and data...")
    model = load_precision_aware_model()
    _, val_data = load_data()
    print(f"Loaded {len(val_data)} cameras from the validation set.")

    if num_examples > len(val_data):
        num_examples = len(val_data)
        print(f"Warning: Number of examples exceeds available validation data. Using {num_examples} examples.")

    sample_cameras = random.sample(val_data, num_examples)

    fig, axes = plt.subplots(num_examples, 1, figsize=(10, 4 * num_examples), sharex=True)
    if num_examples == 1:
        axes = [axes] # Make it iterable if only one
    
    print(f"Generating predictions for {num_examples} sample cameras from the validation set using {data_fraction:.0%} of data...")
    for i, camera_info in enumerate(sample_cameras):
        ax = axes[i]
        
        # Load data and get ground truth
        camera_df = pd.read_parquet(camera_info['file_path'])
        gt_precision = camera_info['tp_ratio_gt']

        # Sample a fraction of the data for prediction
        sampled_df = camera_df.sample(frac=data_fraction, random_state=42) if data_fraction < 1.0 else camera_df
        sample_size = len(sampled_df)

        # Get model prediction
        _, _, weights, locations, scales = predict_densities_and_ratio_precision_aware(model, sampled_df, sample_size)
        
        # Plot predicted distribution
        x = np.linspace(0, 1, 1000)
        pdf = mixture_logistic_pdf_numpy(x, weights, locations, scales)
        
        sns.lineplot(x=x, y=pdf, ax=ax, label="Predicted Distribution")
        
        # Plot ground truth
        ax.axvline(gt_precision, color='r', linestyle='--', label=f"Ground Truth: {gt_precision:.3f}")
        
        # Calculate predicted mean
        predicted_mean = np.sum(weights * locations)
        ax.axvline(predicted_mean, color='g', linestyle=':', label=f"Predicted Mean: {predicted_mean:.3f}")

        ax.set_title(f"Camera: {camera_info['file_path'].split('/')[-1]} (using {data_fraction:.0%} of data)")
        ax.set_xlabel("TP Ratio (Precision)")
        ax.set_ylabel("Probability Density")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(0, 0.1)

    plt.tight_layout()
    output_path = "plots/prediction_examples.png"
    plt.savefig(output_path)
    print(f"✅ Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_predictions(num_examples=5, data_fraction=0.3) 