#!/bin/bash

# env
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd):$PYTHONPATH 

# Hyperparams
INPUT_DIM=784
HIDDEN_DIM=200
LATENT_DIM=20
EPOCHS=30
DEVICE="cuda"
BATCH_SIZE=32
LR=3e-4

# Run
python scripts/train.py \
  --input_dim $INPUT_DIM \
  --hidden_dim $HIDDEN_DIM \
  --latent_dim $LATENT_DIM \
  --lr $LR \
  --batch_size $BATCH_SIZE \
  --num_train_epochs $EPOCHS \
  --device $DEVICE \