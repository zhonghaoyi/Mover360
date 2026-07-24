#!/bin/bash
# Train Mover360 (LoRA adaptation of Flux2-Klein-4B-Base with auxiliary depth).
#
# Usage: ./train.sh [run_name] [data_dir]
#   run_name  experiment name; checkpoints and logs go to logs/<run_name>/
#   data_dir  root of the UE5 training data (default: data/UE5_data)
#
# Tip: set WANDB_MODE=offline to train without a Weights & Biases account.

set -e

RUN_NAME=${1:-mover360_flux2_depth}
DATA_DIR=${2:-data/UE5_data}

WANDB_NAME="$RUN_NAME" WANDB_RUN_ID="$RUN_NAME" \
python -u main.py fit \
    --data=Mover360_Base \
    --model=Mover360_depth \
    --data.init_args.data_dir="$DATA_DIR" \
    --data.init_args.inpaint_data_dir="$DATA_DIR"
