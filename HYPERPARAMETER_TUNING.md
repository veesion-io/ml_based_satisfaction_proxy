# Hyperparameter Tuning for Precision-Aware Training

This directory contains tools for automatically finding the best hyperparameters for your precision-aware model using Bayesian optimization.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_tuning.txt
```

### 2. Run Hyperparameter Tuning

```bash
# Basic tuning with 50 trials
python hyperparameter_tuning.py

# Extended tuning with more trials and longer evaluation
python hyperparameter_tuning.py --n_trials 100 --tuning_epochs 15

# Quick tuning for testing
python hyperparameter_tuning.py --n_trials 20 --tuning_epochs 5
```

### 3. Analyze Results

```bash
# Analyze the latest tuning run
python analyze_tuning_results.py hyperparameter_studies/precision_aware_tuning_YYYYMMDD_HHMMSS.db
```

### 4. Train with Best Parameters

The tuning script will output a command with the best parameters found:

```bash
python precision_aware_training.py \
  --learning_rate 0.001234 \
  --batch_size 8 \
  --phi_dim 256 \
  --num_heads 8 \
  --dropout_rate 0.15 \
  --weight_decay 0.0001 \
  --density_weight 1.2 \
  --distribution_weight 2.5 \
  --precision_weight 1.1 \
  --epochs 50
```

## What Gets Optimized

The tuning script optimizes these hyperparameters:

### Model Architecture
- **phi_dim**: Embedding dimension (128, 208, 256, 320, 384)
- **num_heads**: Number of attention heads (4, 7, 8, 16)
- **dropout_rate**: Dropout probability (0.0 - 0.5)

### Training
- **learning_rate**: Learning rate (1e-5 - 1e-2, log scale)
- **batch_size**: Batch size (2, 4, 8, 16)
- **weight_decay**: L2 regularization (1e-6 - 1e-2, log scale)

### Loss Weights
- **density_weight**: Weight for density loss (0.1 - 3.0)
- **distribution_weight**: Weight for distribution loss (0.1 - 5.0)
- **precision_weight**: Weight for precision loss (0.1 - 3.0)

## Optimization Strategy

The tuning uses **Optuna** with:
- **Bayesian optimization** for efficient parameter search
- **Median pruning** to stop unpromising trials early
- **Early stopping** (3 epochs patience) to speed up evaluation
- **Validation precision loss** as the objective to minimize

## Output Files

### During Tuning
- `hyperparameter_studies/`: SQLite databases with trial history
- Progress displayed in real-time with trial results

### After Tuning
- `tuning_results/best_params_TIMESTAMP.json`: Best parameters found
- `tuning_results/best_training_command.sh`: Ready-to-run training command

### After Analysis
- `tuning_analysis/optimization_overview.png`: Optimization visualizations
- `tuning_analysis/parameter_correlations.png`: Parameter correlation heatmap
- `tuning_analysis/parallel_coordinates.html`: Interactive parameter exploration
- `tuning_analysis/parameter_slices.html`: Interactive parameter slices
- `tuning_analysis/analysis_summary.json`: Detailed statistics
- `tuning_analysis/top_10_trials.csv`: Best performing trials

## Command Line Options

### hyperparameter_tuning.py
```bash
--n_trials 50           # Number of optimization trials
--tuning_epochs 10      # Epochs per trial (fewer = faster)
--timeout 0             # Max time in seconds (0 = no limit)
```

### analyze_tuning_results.py
```bash
study_path              # Path to .db file
--output_dir analysis   # Output directory for results
```

## Tips for Better Tuning

### For Quick Experimentation
```bash
python hyperparameter_tuning.py --n_trials 20 --tuning_epochs 5
```

### For Production Models
```bash
python hyperparameter_tuning.py --n_trials 100 --tuning_epochs 20
```

### For Overnight Runs
```bash
python hyperparameter_tuning.py --n_trials 200 --tuning_epochs 15 --timeout 28800  # 8 hours
```

## Understanding Results

### Key Metrics
- **Objective Value**: Validation precision loss (lower is better)
- **Parameter Importance**: Which parameters matter most
- **Convergence**: How quickly optimization finds good parameters

### What to Look For
- **Stable convergence**: Objective value should stabilize
- **Parameter clustering**: Good parameters often cluster together
- **Trade-offs**: Balance between model complexity and performance

## Resuming Interrupted Tuning

If tuning is interrupted, you can resume by running the same command again. Optuna automatically loads the existing study and continues optimization.

## Advanced Usage

### Custom Parameter Ranges

Edit `hyperparameter_tuning.py` to modify the parameter search spaces:

```python
# Example: Wider learning rate range
learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True)

# Example: More batch size options
batch_size = trial.suggest_categorical('batch_size', [1, 2, 4, 8, 16, 32])
```

### Multi-Objective Optimization

Modify the objective function to optimize multiple metrics:

```python
return best_val_precision_loss, model_complexity_penalty
```

### Conditional Parameters

Add parameter dependencies:

```python
if trial.suggest_categorical('use_large_model', [True, False]):
    phi_dim = trial.suggest_categorical('phi_dim', [512, 768, 1024])
else:
    phi_dim = trial.suggest_categorical('phi_dim', [128, 256, 384])
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch_size options or phi_dim ranges
2. **Slow tuning**: Reduce tuning_epochs or use fewer trials
3. **No improvement**: Increase n_trials or expand parameter ranges
4. **Import errors**: Install requirements with `pip install -r requirements_tuning.txt`

### Performance Tips

- Use fewer `tuning_epochs` for initial exploration (5-10)
- Use more `tuning_epochs` for final optimization (15-25)
- Monitor GPU utilization and adjust batch sizes accordingly
- Use `timeout` for overnight runs to prevent infinite execution 