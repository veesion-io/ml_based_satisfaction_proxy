#!/usr/bin/env python3
"""
Summary and reporting utilities for convergence analysis
"""

import numpy as np

def print_analysis_header():
    """Print analysis header"""
    print("\n" + "="*70)
    print("📈 TP RATIO PREDICTION ERROR SUMMARY (ACROSS CAMERAS)")
    print("="*70)

def compute_overall_performance_stats(results_df):
    """Compute overall performance statistics"""
    best_error = results_df['mean_avg_precision_error'].min()
    worst_error = results_df['mean_avg_precision_error'].max()
    best_prop = results_df.loc[results_df['mean_avg_precision_error'].idxmin(), 'proportion']
    worst_prop = results_df.loc[results_df['mean_avg_precision_error'].idxmax(), 'proportion']
    
    return {
        'best_error': best_error,
        'worst_error': worst_error,
        'best_prop': best_prop,
        'worst_prop': worst_prop,
        'improvement': worst_error - best_error,
        'relative_gain': ((worst_error - best_error) / worst_error * 100)
    }

def print_overall_performance(stats):
    """Print overall performance statistics"""
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   Best Error:       {stats['best_error']:.4f} (at {stats['best_prop']:.1%} data)")
    print(f"   Worst Error:      {stats['worst_error']:.4f} (at {stats['worst_prop']:.1%} data)")
    print(f"   Improvement:      {stats['improvement']:.4f} reduction")
    print(f"   Relative Gain:    {stats['relative_gain']:.1f}%")

def compute_proportion_analysis(results_df):
    """Compute analysis by data proportion ranges"""
    small_props = results_df[results_df['proportion'] <= 0.10]
    medium_props = results_df[(results_df['proportion'] > 0.10) & (results_df['proportion'] <= 0.50)]
    large_props = results_df[results_df['proportion'] > 0.50]
    
    return {
        'small': small_props['mean_avg_precision_error'].mean() if not small_props.empty else None,
        'medium': medium_props['mean_avg_precision_error'].mean() if not medium_props.empty else None,
        'large': large_props['mean_avg_precision_error'].mean() if not large_props.empty else None
    }

def print_proportion_analysis(prop_stats):
    """Print data proportion analysis"""
    print(f"\n📊 DATA PROPORTION ANALYSIS:")
    if prop_stats['small'] is not None:
        print(f"   Small (≤10%):     {prop_stats['small']:.4f} avg error")
    if prop_stats['medium'] is not None:
        print(f"   Medium (10-50%):  {prop_stats['medium']:.4f} avg error")
    if prop_stats['large'] is not None:
        print(f"   Large (>50%):     {prop_stats['large']:.4f} avg error")

def print_error_targets(results_df):
    """Print error target analysis"""
    print(f"\n🏆 ERROR TARGETS:")
    targets = [0.01, 0.02, 0.03, 0.05]
    for target in targets:
        target_reached = results_df[results_df['mean_avg_precision_error'] <= target]
        if not target_reached.empty:
            min_prop = target_reached['proportion'].min()
            print(f"   {target:.1%} Error:      {min_prop:.1%} data needed")
        else:
            print(f"   {target:.1%} Error:      Not achieved")

def compute_convergence_pattern(results_df):
    """Compute convergence pattern analysis"""
    if len(results_df) < 2:
        return None
    
    first_half = results_df.head(len(results_df)//2)
    second_half = results_df.tail(len(results_df)//2)
    
    first_avg = first_half['mean_avg_precision_error'].mean()
    second_avg = second_half['mean_avg_precision_error'].mean()
    improvement = first_avg - second_avg
    
    return {
        'first_avg': first_avg,
        'second_avg': second_avg,
        'improvement': improvement,
        'improvement_pct': improvement/first_avg*100,
        'is_converging': improvement > 0
    }

def print_convergence_pattern(conv_stats):
    """Print convergence pattern analysis"""
    print(f"\n📈 CONVERGENCE PATTERN:")
    
    if conv_stats is None:
        print("   Insufficient data for convergence analysis")
        return
    
    print(f"   Early proportions: {conv_stats['first_avg']:.4f} avg error")
    print(f"   Later proportions: {conv_stats['second_avg']:.4f} avg error")
    print(f"   Improvement:       {conv_stats['improvement']:.4f} ({conv_stats['improvement_pct']:.1f}%)")
    
    if conv_stats['is_converging']:
        print(f"   ✅ Clear convergence: error decreases with more data")
    else:
        print(f"   ⚠️  Plateau/degradation: error may not improve with more data")

def print_sample_size_info(results_df):
    """Print sample size information"""
    print(f"\n📏 SAMPLE SIZE INFO:")
    print(f"   Min sample size:   {results_df['mean_sample_size'].min():.0f} alerts")
    print(f"   Max sample size:   {results_df['mean_sample_size'].max():.0f} alerts")
    print(f"   Average cameras:   {results_df['n_cameras'].mean():.0f} per proportion")

def compute_uncertainty_analysis(results_df):
    """Compute TRUE distribution uncertainty analysis"""
    initial_true_uncertainty = results_df['true_avg_ci_width'].iloc[0]
    final_true_uncertainty = results_df['true_avg_ci_width'].iloc[-1]
    true_uncertainty_reduction = ((initial_true_uncertainty - final_true_uncertainty) / initial_true_uncertainty * 100)
    
    return {
        'initial': initial_true_uncertainty,
        'final': final_true_uncertainty,
        'reduction': true_uncertainty_reduction,
        'mean_std': results_df['std_avg_precision_pred'].mean(),
        'mean_iqr': results_df['true_avg_iqr_width'].mean(),
        'n_samples': results_df['n_monte_carlo_samples'].iloc[0],
        'mean_individual_ci': results_df['mean_ci_width'].mean(),
        'ratio': results_df['true_avg_ci_width'].mean() / results_df['mean_ci_width'].mean()
    }

def print_uncertainty_analysis(uncertainty_stats):
    """Print TRUE distribution uncertainty analysis"""
    print(f"\n🎯 TRUE DISTRIBUTION OF AVERAGES ANALYSIS:")
    print(f"   Initial TRUE Uncertainty:  {uncertainty_stats['initial']:.4f} (95% CI width of avg distribution)")
    print(f"   Final TRUE Uncertainty:    {uncertainty_stats['final']:.4f} (95% CI width of avg distribution)")
    print(f"   TRUE Uncertainty Reduction: {uncertainty_stats['reduction']:.1f}%")
    
    print(f"\n   Distribution of Averages Metrics:")
    print(f"   Mean Std of Averages:     {uncertainty_stats['mean_std']:.4f}")
    print(f"   Mean TRUE IQR Width:      {uncertainty_stats['mean_iqr']:.4f}")
    print(f"   Monte Carlo Samples:      {uncertainty_stats['n_samples']:,}")
    
    print(f"\n   Comparison with Individual Camera Stats:")
    print(f"   Mean Individual Camera CI: {uncertainty_stats['mean_individual_ci']:.4f}")
    print(f"   Ratio (True/Individual):   {uncertainty_stats['ratio']:.3f}")

def compute_correlation_analysis(results_df):
    """Compute correlation between uncertainty and error"""
    if len(results_df) <= 2:
        return None
    
    corr = np.corrcoef(results_df['true_avg_ci_width'], results_df['mean_avg_precision_error'])[0,1]
    return corr

def print_correlation_analysis(correlation):
    """Print uncertainty-error correlation analysis"""
    if correlation is None:
        return
    
    print(f"\n   📊 True Uncertainty-Error Correlation: {correlation:.3f}")
    if correlation > 0.5:
        print(f"   ✅ High correlation: true distribution uncertainty tracks prediction errors well!")
    elif correlation > 0.2:
        print(f"   ⚠️  Moderate correlation: some alignment between true uncertainty and errors")
    else:
        print(f"   ❌ Low correlation: true uncertainty may not reflect prediction quality")

def print_analysis_footer():
    """Print analysis footer"""
    print("\n" + "="*70)

def print_average_precision_summary(results_df):
    """Print average precision convergence summary statistics"""
    
    print_analysis_header()
    
    # Overall performance
    overall_stats = compute_overall_performance_stats(results_df)
    print_overall_performance(overall_stats)
    
    # Data proportion analysis
    prop_stats = compute_proportion_analysis(results_df)
    print_proportion_analysis(prop_stats)
    
    # Error targets
    print_error_targets(results_df)
    
    # Convergence analysis
    conv_stats = compute_convergence_pattern(results_df)
    print_convergence_pattern(conv_stats)
    
    # Sample size info
    print_sample_size_info(results_df)
    
    # TRUE distribution uncertainty analysis
    uncertainty_stats = compute_uncertainty_analysis(results_df)
    print_uncertainty_analysis(uncertainty_stats)
    
    # Correlation analysis
    correlation = compute_correlation_analysis(results_df)
    print_correlation_analysis(correlation)
    
    print_analysis_footer() 