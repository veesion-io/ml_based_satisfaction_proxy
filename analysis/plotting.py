#!/usr/bin/env python3
"""
Plotting utilities for convergence analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def get_color_palette():
    """Define a beautiful color palette for plots"""
    return {
        'median': '#FF6B6B',      # Bright red for median (hottest)
        'q25_75': '#4ECDC4',      # Teal for IQR
        'q10_90': '#45B7D1',      # Blue for 10-90%
        'q05_95': '#96CEB4',      # Light green for 5-95%
        'mean': '#FFA726',        # Orange for mean
        'gt': '#2E8B57',          # Sea green for ground truth
        'background': '#F8F9FA'   # Light background
    }

def setup_plot_style():
    """Set up the plot style and create subplots"""
    plt.style.use('default')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    return fig, ax1, ax2, ax3

def plot_precision_quantiles(ax, results_df, colors):
    """Plot the average precision quantiles (Plot 1)"""
    x = results_df['proportion'] * 100  # Convert to percentage
    
    # Fill areas from outside to inside for beautiful layering effect
    ax.fill_between(x, results_df['q05_avg_precision_pred'], results_df['q95_avg_precision_pred'], 
                     alpha=0.2, color=colors['q05_95'], label='5-95th Percentile')
    ax.fill_between(x, results_df['q10_avg_precision_pred'], results_df['q90_avg_precision_pred'], 
                     alpha=0.3, color=colors['q10_90'], label='10-90th Percentile')
    ax.fill_between(x, results_df['q25_avg_precision_pred'], results_df['q75_avg_precision_pred'], 
                     alpha=0.4, color=colors['q25_75'], label='25-75th Percentile (IQR)')
    
    # Plot the median line as the "hottest" (most prominent)
    ax.plot(x, results_df['median_avg_precision_pred'], linewidth=4, 
             color=colors['median'], label='Median Prediction', 
             alpha=0.95, zorder=10)
    
    # Add markers for all data points to ensure visibility
    ax.scatter(x, results_df['median_avg_precision_pred'], 
                s=80, color='white', edgecolors=colors['median'], 
                linewidth=2, zorder=11, alpha=0.9)
    
    # Add mean line for comparison (more subtle)
    ax.plot(x, results_df['mean_avg_precision_pred'], '--', linewidth=2, 
             color=colors['mean'], alpha=0.7, label='Mean Prediction', zorder=9)

def add_ground_truth_line(ax, gt_avg_precision, colors):
    """Add ground truth horizontal line"""
    ax.axhline(y=gt_avg_precision, color=colors['gt'], linestyle='-', alpha=0.8, 
                linewidth=3, label=f'Ground Truth ({gt_avg_precision:.4f})', zorder=12)

def style_precision_plot(ax, results_df, gt_avg_precision):
    """Style the precision plot with labels and limits"""
    ax.set_xlabel('Data Proportion (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('TP Ratio Across Cameras', fontsize=14, fontweight='bold')
    ax.set_title('TP Ratio Convergence\n(Predicted vs Ground Truth)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_xlim(0, 105)
    
    # Set y-limits centered around the data range, not starting from 0
    y_min = min(results_df['q05_avg_precision_pred'].min(), gt_avg_precision) * 0.98
    y_max = max(results_df['q95_avg_precision_pred'].max(), gt_avg_precision) * 1.02
    ax.set_ylim(y_min, y_max)

def plot_error_quantiles(ax, results_df, colors):
    """Plot the error quantiles (Plot 2)"""
    x = results_df['proportion'] * 100
    
    # Fill areas from outside to inside for beautiful layering effect
    ax.fill_between(x, results_df['q05_avg_precision_error'], results_df['q95_avg_precision_error'], 
                     alpha=0.25, color=colors['q05_95'], label='5-95th Percentile', zorder=1)
    ax.fill_between(x, results_df['q10_avg_precision_error'], results_df['q90_avg_precision_error'], 
                     alpha=0.35, color=colors['q10_90'], label='10-90th Percentile', zorder=2)
    ax.fill_between(x, results_df['q25_avg_precision_error'], results_df['q75_avg_precision_error'], 
                     alpha=0.45, color=colors['q25_75'], label='25-75th Percentile (IQR)', zorder=3)
    
    # Plot the median line as the "hottest" (most prominent)
    ax.plot(x, results_df['median_avg_precision_error'], linewidth=4, 
             color=colors['median'], label='Median Error', 
             alpha=0.95, zorder=10)
    
    # Add markers for all data points to ensure visibility
    ax.scatter(x, results_df['median_avg_precision_error'], 
                s=80, color='white', edgecolors=colors['median'], 
                linewidth=2, zorder=11, alpha=0.9)
    
    # Add mean line for comparison (more subtle)
    ax.plot(x, results_df['mean_avg_precision_error'], '--', linewidth=2, 
             color=colors['mean'], alpha=0.7, label='Mean Error', zorder=9)

def add_target_lines(ax):
    """Add target lines for error thresholds"""
    ax.axhline(y=0.01, color='green', linestyle=':', alpha=0.6, linewidth=2, label='1% Target')
    ax.axhline(y=0.02, color='orange', linestyle=':', alpha=0.6, linewidth=2, label='2% Target')
    ax.axhline(y=0.05, color='red', linestyle=':', alpha=0.6, linewidth=2, label='5% Threshold')

def style_error_plot(ax, results_df):
    """Style the error plot with labels and limits"""
    ax.set_xlabel('Data Proportion (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('|Avg(Predicted TP Ratio) - Avg(GT TP Ratio)| Error', fontsize=14, fontweight='bold')
    ax.set_title('TP Ratio Error Distribution\n(Quantiles of Prediction Errors)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, framealpha=0.9, loc='upper right')
    ax.set_xlim(0, 105)
    
    # Set y-limits for error plot centered around data, not starting from 0
    error_max = results_df['q95_avg_precision_error'].max() * 1.1
    ax.set_ylim(0, error_max)

def plot_uncertainty_analysis(ax, results_df, colors):
    """Plot the uncertainty analysis (Plot 3)"""
    x = results_df['proportion'] * 100
    
    ax.fill_between(x, 
                     results_df['true_avg_ci_width'] * 0.9, 
                     results_df['true_avg_ci_width'] * 1.1, 
                     alpha=0.3, color=colors['q25_75'], label='True Avg CI Width ± 10%')
    
    ax.plot(x, results_df['true_avg_ci_width'], linewidth=3, 
             color=colors['median'], label='True Avg CI Width (95%)', 
             alpha=0.9, zorder=10)
    
    ax.plot(x, results_df['true_avg_iqr_width'], '--', linewidth=2, 
             color=colors['mean'], alpha=0.8, label='True Avg IQR Width (25-75%)', zorder=9)
    
    ax.plot(x, results_df['std_avg_precision_pred'], ':', linewidth=2, 
             color=colors['q10_90'], alpha=0.8, label='Std of Average Distribution', zorder=8)
    
    # Individual camera uncertainty for comparison (thinner line)
    ax.plot(x, results_df['mean_ci_width'], '-.', linewidth=1, 
             color='gray', alpha=0.6, label='Mean Individual Camera CI', zorder=7)
    
    # Add markers for key points
    ax.scatter(x, results_df['true_avg_ci_width'], 
                s=60, color='white', edgecolors=colors['median'], 
                linewidth=2, zorder=11, alpha=0.9)

def style_uncertainty_plot(ax, results_df):
    """Style the uncertainty plot with labels and limits"""
    ax.set_xlabel('Data Proportion (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Distribution Width Metrics', fontsize=14, fontweight='bold')
    ax.set_title('TRUE Distribution of Averages Width\n(Proper Statistical Combination)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xlim(0, 105)
    
    # Set y-limits for uncertainty plot
    uncertainty_max = max(results_df['true_avg_ci_width'].max(), results_df['true_avg_iqr_width'].max()) * 1.1
    ax.set_ylim(0, uncertainty_max)

def add_annotations(ax1, ax3, results_df, gt_avg_precision, colors):
    """Add annotations to key plots"""
    # Annotation for precision plot
    final_median_pred = results_df['median_avg_precision_pred'].iloc[-1]
    final_error = abs(final_median_pred - gt_avg_precision)
    
    ax1.annotate(f'Final Error: {final_error:.4f}\nGT: {gt_avg_precision:.4f}\nPred: {final_median_pred:.4f}',
                xy=(100, final_median_pred),
                xytext=(70, final_median_pred + 0.01),
                arrowprops=dict(arrowstyle='->', color=colors['median'], lw=2),
                fontsize=10, ha='left', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                         edgecolor=colors['median'], alpha=0.9))
    
    # Annotation for uncertainty plot
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

def apply_final_styling(axes):
    """Apply final styling to all axes"""
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.tick_params(labelsize=12)

def create_beautiful_quantile_plot(results_df, gt_avg_precision):
    """Create beautiful quantile plot with filled areas, uncertainty analysis, and prominent median"""
    
    # Setup
    fig, ax1, ax2, ax3 = setup_plot_style()
    colors = get_color_palette()
    
    # Plot 1: Average Precision Quantiles
    plot_precision_quantiles(ax1, results_df, colors)
    add_ground_truth_line(ax1, gt_avg_precision, colors)
    style_precision_plot(ax1, results_df, gt_avg_precision)
    
    # Plot 2: Error quantiles
    plot_error_quantiles(ax2, results_df, colors)
    add_target_lines(ax2)
    style_error_plot(ax2, results_df)
    
    # Plot 3: Uncertainty Analysis
    plot_uncertainty_analysis(ax3, results_df, colors)
    style_uncertainty_plot(ax3, results_df)
    
    # Add annotations
    add_annotations(ax1, ax3, results_df, gt_avg_precision, colors)
    
    # Final styling
    apply_final_styling([ax1, ax2, ax3])
    
    # Save and show
    plt.tight_layout()
    plt.savefig('tp_ratio_convergence_plot_with_uncertainty.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    print(f"📊 Enhanced plot with uncertainty analysis saved to tp_ratio_convergence_plot_with_uncertainty.png") 