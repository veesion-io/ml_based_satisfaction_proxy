#!/usr/bin/env python3
"""
Hyperparameter Tuning for Model with TP Ratio Prediction
Uses Optuna to optimize hyperparameters for the new architecture
"""

import optuna
import subprocess
import re
import os
import time
import argparse
from pathlib import Path

def objective(trial):
    """Objective function for Optuna optimization"""
    
    # Suggest hyperparameters with updated ranges based on new architecture
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 4, 32, step=2)
    phi_dim = trial.suggest_int('phi_dim', 64, 256, step=16)  # Ensure divisible by num_heads
    num_heads = trial.suggest_int('num_heads', 2, 8)
    epochs = trial.suggest_int('epochs', 15, 25)  # Slightly reduced since we have good baseline
    
    # Ensure phi_dim is divisible by num_heads
    phi_dim = ((phi_dim // num_heads) + 1) * num_heads if phi_dim % num_heads != 0 else phi_dim
    
    # GPU assignment for parallel trials
    gpu_id = trial.number % 2  # Distribute across 2 GPUs
    
    # Set environment variable for GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"\nTrial {trial.number}: GPU {gpu_id}")
    print(f"Params: lr={learning_rate:.2e}, batch={batch_size}, phi={phi_dim}, heads={num_heads}, epochs={epochs}")
    
    try:
        # Run training with suggested hyperparameters
        cmd = [
            'python', 'density_prediction_training_with_ratio.py',
            '--learning_rate', str(learning_rate),
            '--batch_size', str(batch_size),
            '--phi_dim', str(phi_dim),
            '--num_heads', str(num_heads),
            '--epochs', str(epochs)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            print(f"Training failed: {result.stderr}")
            return -1.0  # Return bad score for failed runs
        
        # Extract final R² score from output
        output = result.stdout
        
        # Look for the final R² score line
        r2_pattern = r"Final R2 Score: ([\d\.-]+)"
        match = re.search(r2_pattern, output)
        
        if match:
            r2_score = float(match.group(1))
            print(f"Trial {trial.number} R² Score: {r2_score:.4f}")
            
            # Also extract individual scores for analysis
            tp_pattern = r"Final TP R2: ([\d\.-]+)"
            fp_pattern = r"Final FP R2: ([\d\.-]+)"
            ratio_pattern = r"Final Ratio R2: ([\d\.-]+)"
            
            tp_match = re.search(tp_pattern, output)
            fp_match = re.search(fp_pattern, output)
            ratio_match = re.search(ratio_pattern, output)
            
            if tp_match and fp_match and ratio_match:
                tp_r2 = float(tp_match.group(1))
                fp_r2 = float(fp_match.group(1))
                ratio_r2 = float(ratio_match.group(1))
                
                print(f"  TP R²: {tp_r2:.4f}, FP R²: {fp_r2:.4f}, Ratio R²: {ratio_r2:.4f}")
                
                # Store additional metrics
                trial.set_user_attr('tp_r2', tp_r2)
                trial.set_user_attr('fp_r2', fp_r2)
                trial.set_user_attr('ratio_r2', ratio_r2)
            
            return r2_score
        else:
            print(f"Could not extract R² score from output")
            return -1.0
            
    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} timed out")
        return -1.0
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return -1.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=150, help='Number of trials')
    parser.add_argument('--n_jobs', type=int, default=8, help='Number of parallel jobs')
    parser.add_argument('--study_name', type=str, default='model_with_ratio_tuning', help='Study name')
    args = parser.parse_args()
    
    print("🎯 Starting Hyperparameter Tuning for Model with TP Ratio")
    print("=" * 60)
    print(f"Trials: {args.n_trials}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"Study name: {args.study_name}")
    print(f"GPUs available: 2 (will distribute trials)")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name=args.study_name,
        storage=f'sqlite:///{args.study_name}.db',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    print(f"\nStarting optimization...")
    start_time = time.time()
    
    # Run optimization
    study.optimize(
        objective, 
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=7200  # 2 hours total timeout
    )
    
    end_time = time.time()
    
    # Print results
    print("\n" + "=" * 60)
    print("🏆 OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"Time taken: {(end_time - start_time) / 60:.1f} minutes")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Best R² score: {study.best_value:.4f}")
    
    print(f"\n🎯 BEST HYPERPARAMETERS:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    # Get additional metrics for best trial
    best_trial = study.best_trial
    if hasattr(best_trial, 'user_attrs'):
        if 'tp_r2' in best_trial.user_attrs:
            print(f"\n📊 BEST TRIAL DETAILED SCORES:")
            print(f"   TP R²: {best_trial.user_attrs['tp_r2']:.4f}")
            print(f"   FP R²: {best_trial.user_attrs['fp_r2']:.4f}")
            print(f"   Ratio R²: {best_trial.user_attrs['ratio_r2']:.4f}")
    
    # Save results
    results_file = f"best_hyperparameters_with_ratio.txt"
    with open(results_file, 'w') as f:
        f.write(f"Best R² Score: {study.best_value:.4f}\n")
        f.write(f"Best Hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
        
        if hasattr(best_trial, 'user_attrs') and 'tp_r2' in best_trial.user_attrs:
            f.write(f"\nDetailed Scores:\n")
            f.write(f"  TP R²: {best_trial.user_attrs['tp_r2']:.4f}\n")
            f.write(f"  FP R²: {best_trial.user_attrs['fp_r2']:.4f}\n")
            f.write(f"  Ratio R²: {best_trial.user_attrs['ratio_r2']:.4f}\n")
    
    print(f"\n✅ Results saved to {results_file}")
    
    # Show top trials
    print(f"\n🥇 TOP 5 TRIALS:")
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)[:5]
    
    for i, trial in enumerate(top_trials):
        if trial.value is not None:
            print(f"{i+1}. R²={trial.value:.4f} | lr={trial.params['learning_rate']:.2e} | "
                  f"batch={trial.params['batch_size']} | phi={trial.params['phi_dim']} | "
                  f"heads={trial.params['num_heads']} | epochs={trial.params['epochs']}")
    
    # Plot optimization history
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Optimization history
        values = [t.value for t in study.trials if t.value is not None]
        ax1.plot(values, 'o-', alpha=0.7)
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Optimization History')
        ax1.grid(True, alpha=0.3)
        
        # Parameter importance (if enough trials)
        if len(study.trials) >= 10:
            try:
                importance = optuna.importance.get_param_importances(study)
                params = list(importance.keys())
                importances = list(importance.values())
                
                ax2.barh(params, importances)
                ax2.set_xlabel('Importance')
                ax2.set_title('Parameter Importance')
            except Exception as e:
                ax2.text(0.5, 0.5, f'Could not compute\nparameter importance:\n{str(e)}', 
                        ha='center', va='center', transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, 'Not enough trials\nfor parameter importance', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.savefig(f'{args.study_name}_optimization.png', dpi=300, bbox_inches='tight')
        print(f"📊 Optimization plots saved to {args.study_name}_optimization.png")
        
    except ImportError:
        print("⚠️  Matplotlib not available for plotting")
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")
    
    print(f"\n🎉 Hyperparameter tuning complete!")

if __name__ == "__main__":
    main() 