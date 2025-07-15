#!/bin/bash

# env
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd):$PYTHONPATH 

# Hyperparams
IMG_SIZE=28
GEN_SAMPLE_SHAPE="(20,1,28,28)"
NUM_LABELS=10
INPUT_CHANNEL=1
TIME_ENCODING_DIM=100
NUM_TIMESTEPS=1000
EPOCHS=10
DEVICE="cuda"
BATCH_SIZE=32
LR=1e-3

# Run
python scripts/train.py \
  --img_size $IMG_SIZE \
  --gen_sample_shape $GEN_SAMPLE_SHAPE \
  --input_channel $INPUT_CHANNEL \
  --time_encoding_dim $TIME_ENCODING_DIM \
  --num_labels $NUM_LABELS \
  --lr $LR \
  --batch_size $BATCH_SIZE \
  --num_train_epochs $EPOCHS \
  --num_timesteps $NUM_TIMESTEPS \
  --device $DEVICE \