#!/bin/bash
# Evaluate a folder of Mover360 predictions against ground truth
# (FAED, PSNR, SSIM, LPIPS, DINOv3 similarity, FID).
#
# Usage: ./test.sh <task> <result_dir> [test_data_dir] [run_name]
#   task           move | add | remove
#   result_dir     folder with predicted panoramas (e.g. logs/<run>/predict)
#   test_data_dir  folder with the ground-truth test tuples
#                  (default: data/UE5_data/Test_all_new)
#   run_name       name for this evaluation run (default: <task>_eval)
#
# Tip: set WANDB_MODE=offline to run without a Weights & Biases account.

set -e

TASK=${1:?usage: ./test.sh <move|add|remove> <result_dir> [test_data_dir] [run_name]}
RESULT_DIR=${2:?usage: ./test.sh <move|add|remove> <result_dir> [test_data_dir] [run_name]}
TEST_DATA_DIR=${3:-data/UE5_data/Test_all_new}
RUN_NAME=${4:-${TASK}_eval}

WANDB_NAME="$RUN_NAME" WANDB_RUN_ID="$RUN_NAME" \
python -u main.py test \
    --data=Mover360_Base \
    --model=models.Mover360_depth.EvalPanoGen.EvalPanoGen \
    --test_function="$TASK" \
    --data.init_args.test_function="$TASK" \
    --data.init_args.predict_data_dir="$TEST_DATA_DIR" \
    --data.init_args.num_workers=0 \
    --data.init_args.val_batch_size=1 \
    --result_dir="$RESULT_DIR" \
    --trainer.accelerator=auto \
    --trainer.devices=1 \
    --trainer.strategy=auto \
    --trainer.precision=32-true
