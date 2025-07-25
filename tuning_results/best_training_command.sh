#!/bin/bash
# Best hyperparameters found by tuning
python precision_aware_training.py \
  --learning_rate 7.55361058619587e-05 \
  --batch_size 8 \
  --phi_dim 256 \
  --num_heads 8 \
  --dropout_rate 0.02409659993009239 \
  --weight_decay 1.0476387568266772e-05 \
  --density_weight 2.192891348615809 \
  --distribution_weight 3.2178410195605887 \
  --precision_weight 0.8380863822441287 \
  --epochs 50
