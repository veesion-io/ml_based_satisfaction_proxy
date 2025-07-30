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
    """Plot precision quantiles with a clear, professional look"""
    x = results_df['proportion'] * 100
    
    # Plot Mean of the TRUE distribution of averages
    ax.plot(x, results_df['mean_avg_precision_pred'], 
            color=colors['mean_line'], label='Mean Predicted TP Ratio (Avg across Cameras)', linewidth=2.5)

    # Plot the TRUE 95% CI of the distribution of AVERAGES
    ax.fill_between(x, results_df['q025_avg_precision_pred'], results_df['q975_avg_precision_pred'], 
                    color=colors['true_ci_fill'], alpha=0.3, label='95% CI of the Average TP Ratio')

    
    # Style the plot
    ax.set_xlabel('Proportion of Data Used per Camera (%)', fontsize=14)
    ax.set_ylabel('TP Ratio Across Cameras')
    ax.set_title('TP Ratio Convergence\n(Predicted vs Ground Truth)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}%'))
    
    # Dynamic y-axis scaling
    y_min = min(results_df['q025_avg_precision_pred'].min(), (results_df['mean_avg_precision_pred'] - (results_df['mean_ci_width'] / 2)).min())
    y_max = max(results_df['q975_avg_precision_pred'].max(), (results_df['mean_avg_precision_pred'] + (results_df['mean_ci_width'] / 2)).max())
    padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - padding, y_max + padding)

def add_ground_truth_line(ax, gt_avg_precision, colors):
    """Add a ground truth line to the plot"""
    ax.axhline(y=gt_avg_precision, color=colors['gt_line'], linestyle='-', alpha=0.8, 
               label=f'Ground Truth TP Ratio ({gt_avg_precision:.4f})')
    ax.legend()

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
    y_min = min(results_df['q025_avg_precision_pred'].min(), gt_avg_precision)
    y_max = max(results_df['q975_avg_precision_pred'].max(), gt_avg_precision)
    padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - padding, y_max + padding)

def plot_error_quantiles(ax, results_df, colors):
    """Plot the error quantiles (Plot 2)"""
    x = results_df['proportion'] * 100
    
    # Plot Mean Absolute Error
    ax.plot(x, results_df['mean_avg_precision_error'], 
            color=colors['mean_line'], label='Mean Absolute Error', linewidth=2.5)

    # Fill between the 25th and 75th percentile of errors
    ax.fill_between(x, results_df['q25_avg_precision_error'], results_df['q75_avg_precision_error'], 
                    color=colors['true_ci_fill'], alpha=0.3, label='25-75th Percentile Error (IQR)')
    
    # Style the plot
    ax.set_xlabel('Proportion of Data Used per Camera (%)', fontsize=14)
    ax.set_ylabel('|Avg(Predicted TP Ratio) - Avg(GT TP Ratio)| Error')
    ax.set_title('TP Ratio Error Distribution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}%'))
    ax.set_ylim(0, results_df['q975_avg_precision_error'].max() * 1.1)

def style_error_plot(ax, results_df):
    """Style the error plot with labels and limits"""
    ax.set_xlabel('Data Proportion (%)')
    ax.set_ylabel('|Avg(Predicted TP Ratio) - Avg(GT TP Ratio)| Error')
    ax.set_title('TP Ratio Error Distribution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}%'))
    ax.set_ylim(0, results_df['q975_avg_precision_error'].max() * 1.1)

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
    """Create a high-quality quantile plot with enhanced styling"""
    
    # Use a professional plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Define a clear and professional color palette
    colors = {
        'mean_line': '#003366',              # Deep blue for the mean line
        'true_ci_fill': '#6699CC',           # Lighter blue for the CI of the average
        'single_camera_ci_fill': '#FFC72C',  # Amber/gold for single camera CI
        'gt_line': '#D40000',                # Strong red for ground truth
        'annotations': '#333333'             # Dark gray for text
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot 1: Average Precision Quantiles
    plot_precision_quantiles(ax1, results_df, colors)
    add_ground_truth_line(ax1, gt_avg_precision_across_cameras, colors)
    
    # Plot 2: Error Metrics
    plot_error_quantiles(ax2, results_df, colors)
    
    # Final styling
    fig.suptitle('TP Ratio Convergence Analysis (Precision-Aware Model)', fontsize=20, weight='bold')
    
    plt.tight_layout()
    output_path = "tp_ratio_convergence_plot_with_uncertainty.png"
    plt.savefig(output_path, dpi=300)
    print(f"📊 Enhanced plot with uncertainty analysis saved to {output_path}") 