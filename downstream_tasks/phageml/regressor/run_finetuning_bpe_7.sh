#!/bin/bash

#conda activate bert24;
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT:$PYTHONPATH

CONFIG_DIR="/home/jovyan/shares/SR003.nfs2/caduseus_artem/phageml/ModernBERT/yamls/phageml/simple_test"

CKPT="base_metavr_bpe_continue"
TASK="simple_test_${CKPT}-positive_dataset_v2_lr_1e-6_200e"
CONFIG="${CONFIG_DIR}/positive_dataset/simple_test_bpe_continue-positive_dataset_v2-lr_1e-6_200e.yaml"


MODERNBERT_HOME="/home/jovyan/shares/SR003.nfs2/caduseus_artem/phageml/ModernBERT" CUDA_VISIBLE_DEVICES=0 python downstream_tasks/phageml/regressor/train_with_accelerate.py \
    --config $CONFIG \
    --output_dir "/home/jovyan/shares/SR003.nfs2/caduseus_artem/phageml/output/${TASK}"
