#!/usr/bin/env python3
"""
Plotting utilities for convergence analysis
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np

def get_color_palette():
    """Define a beautiful and consistent color palette"""
    return {
        'q05_95': '#a9d6e5',
        'q10_90': '#89c2d9',
        'q25_75': '#61a5c2',
        'median': '#2c7da0',
        'mean': '#014f86',
        'gt': '#d90429'
    }

def setup_plot_style():
    """Set up the plot style and create subplots"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    return fig, ax1, ax2, ax3

def plot_precision_quantiles(ax, results_df, colors):
    """Plot the average precision quantiles (Plot 1)"""
    x = results_df['proportion'] * 100
    
    ax.fill_between(x, results_df['q05_avg_precision_pred'], results_df['q95_avg_precision_pred'], 
                     alpha=0.2, color=colors['q05_95'], label='5-95th Percentile')
    ax.fill_between(x, results_df['q25_avg_precision_pred'], results_df['q75_avg_precision_pred'], 
                     alpha=0.4, color=colors['q25_75'], label='25-75th Percentile (IQR)')
    ax.plot(x, results_df['median_avg_precision_pred'], linewidth=2, 
             color=colors['median'], label='Median Prediction')
    ax.plot(x, results_df['mean_avg_precision_pred'], '--', linewidth=2, 
             color=colors['mean'], label='Mean Prediction')

def add_ground_truth_line(ax, gt_avg_precision, colors):
    """Add ground truth horizontal line"""
    ax.axhline(y=gt_avg_precision, color=colors['gt'], linestyle='-', alpha=0.8, 
                linewidth=2, label=f'Ground Truth ({gt_avg_precision:.4f})')

def style_precision_plot(ax, results_df, gt_avg_precision):
    """Style the precision plot with labels and limits"""
    ax.set_xlabel('Data Proportion (%)')
    ax.set_ylabel('TP Ratio Across Cameras')
    ax.set_title('TP Ratio Convergence\n(Predicted vs Ground Truth)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}%'))
    
    # Dynamic y-axis scaling
    y_min = min(results_df['q05_avg_precision_pred'].min(), gt_avg_precision)
    y_max = max(results_df['q95_avg_precision_pred'].max(), gt_avg_precision)
    padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - padding, y_max + padding)

def plot_error_quantiles(ax, results_df, colors):
    """Plot the error quantiles (Plot 2)"""
    x = results_df['proportion'] * 100
    
    ax.fill_between(x, results_df['q05_avg_precision_error'], results_df['q95_avg_precision_error'], 
                     alpha=0.2, color=colors['q05_95'], label='5-95th Percentile')
    ax.fill_between(x, results_df['q25_avg_precision_error'], results_df['q75_avg_precision_error'], 
                     alpha=0.4, color=colors['q25_75'], label='25-75th Percentile (IQR)')
    ax.plot(x, results_df['median_avg_precision_error'], linewidth=2, 
             color=colors['median'], label='Median Error')

def style_error_plot(ax, results_df):
    """Style the error plot with labels and limits"""
    ax.set_xlabel('Data Proportion (%)')
    ax.set_ylabel('|Avg(Predicted TP Ratio) - Avg(GT TP Ratio)| Error')
    ax.set_title('TP Ratio Error Distribution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}%'))
    ax.set_ylim(0, results_df['q95_avg_precision_error'].max() * 1.1)

def plot_uncertainty_analysis(ax, results_df, colors):
    """Plot the uncertainty analysis (Plot 3)"""
    x = results_df['proportion'] * 100
    
    ax.plot(x, results_df['true_avg_ci_width'], linewidth=2, 
             color=colors['median'], label='True Avg CI Width (95%)')
    ax.plot(x, results_df['true_avg_iqr_width'], '--', linewidth=2, 
             color=colors['mean'], label='True Avg IQR Width (25-75%)')

def style_uncertainty_plot(ax):
    """Style the uncertainty plot with labels and limits"""
    ax.set_xlabel('Data Proportion (%)')
    ax.set_ylabel('Distribution Width')
    ax.set_title('TRUE Distribution of Averages Width')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}%'))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)

def create_beautiful_quantile_plot(results_df, gt_avg_precision_across_cameras):
    """Create the three-panel plot for convergence analysis."""
    
    fig, ax1, ax2, ax3 = setup_plot_style()
    colors = get_color_palette()
    
    # Plot 1: Average Precision Quantiles
    plot_precision_quantiles(ax1, results_df, colors)
    add_ground_truth_line(ax1, gt_avg_precision_across_cameras, colors)
    style_precision_plot(ax1, results_df, gt_avg_precision_across_cameras)
    
    # Plot 2: Error quantiles
    plot_error_quantiles(ax2, results_df, colors)
    style_error_plot(ax2, results_df)
    
    # Plot 3: Uncertainty Analysis
    plot_uncertainty_analysis(ax3, results_df, colors)
    style_uncertainty_plot(ax3)
    
    plt.tight_layout()
    output_path = "tp_ratio_convergence_plot_with_uncertainty.png"
    plt.savefig(output_path, dpi=300)
    print(f"📊 Enhanced plot with uncertainty analysis saved to {output_path}") 