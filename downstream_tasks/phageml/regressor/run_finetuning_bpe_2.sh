#!/bin/bash

#conda activate bert24;
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT:$PYTHONPATH

CONFIG_DIR="/home/jovyan/shares/SR003.nfs2/caduseus_artem/phageml/ModernBERT/yamls/phageml/multitest"

CKPT="base_metavr_bpe_continue"
TASK="multitest_${CKPT}-weighted_bins-positive_dataset_v2-backbone_dropout_0.1-corr-lr_1e-6_200e"
CONFIG="${CONFIG_DIR}/multitest_bpe_continue-weighted_bins-positive_dataset_v2-backbone_dropout_0.1-corr-lr_1e-6_200e.yaml"


MODERNBERT_HOME="/home/jovyan/shares/SR003.nfs2/caduseus_artem/phageml/ModernBERT" CUDA_VISIBLE_DEVICES=5 python downstream_tasks/phageml/regressor/train_with_accelerate.py \
    --config $CONFIG \
    --output_dir "/home/jovyan/shares/SR003.nfs2/caduseus_artem/phageml/output/${TASK}"
