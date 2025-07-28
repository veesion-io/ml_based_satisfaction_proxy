#!/bin/bash
# Best hyperparameters found by tuning
python precision_aware_training.py \
  --learning_rate 1.0585213619916665e-05 \
  --batch_size 16 \
  --phi_dim 128 \
  --num_heads 4 \
  --dropout_rate 0.0026883822355923716 \
  --weight_decay 0.0005443717754479964 \
  --distribution_weight 2.321171016305433 \
  --precision_weight 0.0383440437257167 \
  --epochs 50
