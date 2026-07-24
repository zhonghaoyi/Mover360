#!/bin/bash
# Run Mover360 prediction on a folder of editing tuples.
#
# Usage: ./predict.sh <task> <ckpt> [predict_data_dir] [run_name] [repeat]
#   task              move | add | remove
#   ckpt              path to a trained checkpoint (.ckpt)
#   predict_data_dir  folder with input tuples (default: data/UE5_data/Test_all_new)
#   run_name          output name; results go to logs/<run_name>/predict/
#   repeat            samples generated per input (default: 1)
#
# Tip: set WANDB_MODE=offline to run without a Weights & Biases account.

set -e

TASK=${1:?usage: ./predict.sh <move|add|remove> <ckpt> [predict_data_dir] [run_name] [repeat]}
CKPT=${2:?usage: ./predict.sh <move|add|remove> <ckpt> [predict_data_dir] [run_name] [repeat]}
PREDICT_DATA_DIR=${3:-data/UE5_data/Test_all_new}
RUN_NAME=${4:-mover360_predict_${TASK}}
REPEAT_PREDICT=${5:-1}

WANDB_NAME="$RUN_NAME" WANDB_RUN_ID="$RUN_NAME" \
python -u main.py predict \
    --data=Mover360_Base \
    --model=Mover360_depth \
    --test_function="$TASK" \
    --data.init_args.predict_data_dir="$PREDICT_DATA_DIR" \
    --data.init_args.repeat_predict="$REPEAT_PREDICT" \
    --ckpt "$CKPT"
