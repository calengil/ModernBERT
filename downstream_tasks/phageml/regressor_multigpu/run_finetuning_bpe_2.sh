#!/bin/bash
#WORK_DIR="/home/jovyan/shares/SR003.nfs2/caduseus_artem/phageml/ModernBERT"
OUTPUT_DIR="output/"

mkdir -p $OUTPUT_DIR
#cd $WORK_DIR
TASK="multitest_bpe_continue-multi-call-weighted_bins-positive_dataset_v2-backbone_dropout_0.01-corr-lr_1e-6_700e"

export PYTHONNOUSERSITE=1
export PYTHONPATH=$WORK_DIR:$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export MODERNBERT_HOME=$WORK_DIR

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # Count the number of GPUs (comma-separated)
    gpu=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    echo "Error: CUDA_VISIBLE_DEVICES is not set!"
    echo "Please run the script like: CUDA_VISIBLE_DEVICES=0,1,2,3 ./script.sh"
    exit 1
fi

CONFIG_DIR="${WORK_DIR}/yamls/phageml/multigpu"
CONFIG="${CONFIG_DIR}/${TASK}.yaml"

torchrun \
    --standalone \
    --nproc_per_node=$gpu \
    downstream_tasks/phageml/regressor_multigpu/train_with_accelerate_multigpu.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR/${TASK}"