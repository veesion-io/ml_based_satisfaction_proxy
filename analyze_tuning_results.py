#!/usr/bin/env python3
"""
Analyze Hyperparameter Tuning Results
Provides visualization and analysis of Optuna optimization results
"""

import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json
from pathlib import Path
import numpy as np

def load_study(study_path):
    """Load Optuna study from SQLite database"""
    storage_url = f"sqlite:///{study_path}"
    
    # Get study name from the database
    study_summaries = optuna.get_all_study_summaries(storage_url)
    if not study_summaries:
        raise ValueError(f"No studies found in {study_path}")
    
    study_name = study_summaries[0].study_name
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    
    return study

def analyze_study(study, output_dir="tuning_analysis"):
    """Analyze and visualize study results"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Study: {study.study_name}")
    print(f"Direction: {study.direction}")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Best value: {study.best_value:.6f}")
    print(f"Best trial: {study.best_trial.number}")
    
    # Create DataFrame from trials
    df = study.trials_dataframe()
    df = df[df['state'] == 'COMPLETE']  # Only completed trials
    
    if len(df) == 0:
        print("No completed trials found!")
        return
    
    print(f"\nCompleted trials: {len(df)}")
    
    # Basic statistics
    print(f"\nObjective Statistics:")
    print(f"  Mean: {df['value'].mean():.6f}")
    print(f"  Std:  {df['value'].std():.6f}")
    print(f"  Min:  {df['value'].min():.6f}")
    print(f"  Max:  {df['value'].max():.6f}")
    
    # Plot optimization history
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(df['number'], df['value'], marker='o', alpha=0.7)
    plt.axhline(y=study.best_value, color='r', linestyle='--', alpha=0.7, label=f'Best: {study.best_value:.6f}')
    plt.xlabel('Trial')
    plt.ylabel('Objective Value')
    plt.title('Optimization History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot parameter importance
    plt.subplot(2, 2, 2)
    try:
        importance = optuna.importance.get_param_importances(study)
        params = list(importance.keys())
        values = list(importance.values())
        
        plt.barh(params, values)
        plt.xlabel('Importance')
        plt.title('Parameter Importance')
        plt.grid(True, alpha=0.3)
    except Exception as e:
        plt.text(0.5, 0.5, f'Parameter importance\nnot available:\n{str(e)}', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Parameter Importance (N/A)')
    
    # Distribution of objective values
    plt.subplot(2, 2, 3)
    plt.hist(df['value'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(x=study.best_value, color='r', linestyle='--', alpha=0.7, label=f'Best: {study.best_value:.6f}')
    plt.xlabel('Objective Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Objective Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Best parameters
    plt.subplot(2, 2, 4)
    best_params = study.best_params
    param_names = list(best_params.keys())
    param_values = [str(v) for v in best_params.values()]
    
    # Create a text plot of best parameters
    plt.text(0.1, 0.9, 'Best Parameters:', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
    
    for i, (name, value) in enumerate(zip(param_names, param_values)):
        y_pos = 0.8 - i * 0.08
        plt.text(0.1, y_pos, f'{name}: {value}', fontsize=10, transform=plt.gca().transAxes)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Best Parameters')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/optimization_overview.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Parameter correlation heatmap
    param_cols = [col for col in df.columns if col.startswith('params_')]
    if len(param_cols) > 1:
        plt.figure(figsize=(10, 8))
        
        # Extract numeric parameters only
        numeric_params = {}
        for col in param_cols:
            try:
                numeric_params[col.replace('params_', '')] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
        
        if len(numeric_params) > 1:
            param_df = pd.DataFrame(numeric_params)
            correlation_matrix = param_df.corr()
            
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
            plt.title('Parameter Correlation Matrix')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/parameter_correlations.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    # Parallel coordinate plot
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(f"{output_dir}/parallel_coordinates.html")
        print(f"Parallel coordinates plot saved to {output_dir}/parallel_coordinates.html")
    except Exception as e:
        print(f"Could not create parallel coordinates plot: {e}")
    
    # Parameter slice plot
    try:
        fig = optuna.visualization.plot_slice(study)
        fig.write_html(f"{output_dir}/parameter_slices.html")
        print(f"Parameter slices plot saved to {output_dir}/parameter_slices.html")
    except Exception as e:
        print(f"Could not create parameter slices plot: {e}")
    
    # Save detailed results
    results_summary = {
        'study_name': study.study_name,
        'direction': str(study.direction),
        'n_trials': len(study.trials),
        'n_completed': len(df),
        'best_value': study.best_value,
        'best_trial_number': study.best_trial.number,
        'best_params': study.best_params,
        'objective_stats': {
            'mean': df['value'].mean(),
            'std': df['value'].std(),
            'min': df['value'].min(),
            'max': df['value'].max(),
            'median': df['value'].median()
        }
    }
    
    with open(f"{output_dir}/analysis_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save top 10 trials
    top_trials = df.nsmallest(10, 'value')[['number', 'value'] + param_cols]
    top_trials.to_csv(f"{output_dir}/top_10_trials.csv", index=False)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/")
    print(f"  - optimization_overview.png: Main visualization")
    print(f"  - analysis_summary.json: Summary statistics")
    print(f"  - top_10_trials.csv: Best trials")
    
    return results_summary

def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter tuning results")
    parser.add_argument('study_path', type=str, 
                       help='Path to the Optuna study database (.db file)')
    parser.add_argument('--output_dir', type=str, default='tuning_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.study_path):
        print(f"Error: Study database not found at {args.study_path}")
        print("\nAvailable studies:")
        study_dir = "hyperparameter_studies"
        if os.path.exists(study_dir):
            for file in os.listdir(study_dir):
                if file.endswith('.db'):
                    print(f"  {os.path.join(study_dir, file)}")
        else:
            print("  No hyperparameter_studies directory found")
        return
    
    print("="*80)
    print("HYPERPARAMETER TUNING RESULTS ANALYSIS")
    print("="*80)
    
    try:
        study = load_study(args.study_path)
        analyze_study(study, args.output_dir)
    except Exception as e:
        print(f"Error analyzing study: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 