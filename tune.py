import optuna
import subprocess
import re
import os
import torch

def objective(trial):
    """
    Defines a single trial for Optuna to optimize.
    """
    # 1. Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 256)
    phi_dim = trial.suggest_int('phi_dim', 32, 256)
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Params: lr={learning_rate:.6f}, bs={batch_size}, phi_dim={phi_dim}, heads={num_heads}")

    # --- GPU Assignment for Parallel Trials ---
    # Assign a GPU to this trial based on the trial number
    n_gpus = torch.cuda.device_count()
    gpu_id = trial.number % n_gpus
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"Trial {trial.number} assigned to GPU {gpu_id}")

    # 2. Construct the command to run the training script
    command = [
        'source', 'satisfaction_proxy_env/bin/activate', '&&',
        'python', 'density_prediction_training.py',
        '--learning_rate', str(learning_rate),
        '--batch_size', str(batch_size),
        '--phi_dim', str(phi_dim),
        '--num_heads', str(num_heads)
        # We REMOVED --no-gpu to allow the script to use the assigned GPU
    ]
    
    command_str = " ".join(command)
    
    try:
        # 3. Execute the training script in its assigned environment
        result = subprocess.run(
            command_str,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            executable='/bin/bash',
            env=env # Pass the modified environment with the correct GPU
        )
        
        output = result.stdout
        print(output)

        # 4. Parse the final R² score
        match = re.search(r"Final R2 Score: ([\-0-9\.]+)", output)
        if match:
            r2_score = float(match.group(1))
            print(f"Trial {trial.number} on GPU {gpu_id} finished with R² score: {r2_score}")
            return r2_score
        else:
            print("Error: Could not find R2 score in training output.")
            return -1.0

    except subprocess.CalledProcessError as e:
        print(f"Error during training subprocess for trial {trial.number} on GPU {gpu_id}:")
        print(e.stderr)
        return -1.0

def main():
    """
    Main function to run the Optuna study.
    """
    study = optuna.create_study(
        direction='maximize',
        study_name='satisfaction_proxy_tuning_gpu_parallel'
    )
    
    # Run trials in parallel, distributing evenly across available GPUs
    n_trials = 200
    n_parallel_jobs = 16
    
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("Warning: No GPUs found. Running trials on CPU.")
    else:
        print(f"Found {n_gpus} GPUs. Distributing {n_parallel_jobs} parallel trials.")

    print(f"Starting Optuna study with {n_trials} trials...")
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_parallel_jobs)
    
    print("\n--- Tuning Complete ---")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best R² score: {study.best_value}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 