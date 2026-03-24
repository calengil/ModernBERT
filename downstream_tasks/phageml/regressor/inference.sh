#!/bin/bash

PYTHONNOUSERSITE=1 
#export PYTHONPATH=/home/jovyan/shares/SR003.nfs2/caduseus_artem/phageml/ModernBERT:$PYTHONPATH
export PYTHONPATH=/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT:$PYTHONPATH
BASE_DIR="/home/jovyan/shares/SR003.nfs2/caduseus_artem/phageml/output"

CKPT="base_metavr_bpe_continue"
TASK="multitest_${CKPT}-weighted_bins-positive_dataset_v2-backbone_dropout_0.1-corr-lr_1e-6_200e"
#CONFIG_DIR="/home/jovyan/shares/SR003.nfs2/caduseus_artem/phageml/ModernBERT/yamls/phageml/multitest"
CONFIG_DIR="${BASE_DIR}/${TASK}"
CONFIG="${CONFIG_DIR}/config.yaml"


#BASE_DIR="/home/jovyan/shares/SR003.nfs2/caduseus_artem/phageml/output"
OUTPUT_DIR="/home/jovyan/shares/SR003.nfs2/caduseus_artem/phageml/output/inference"
required_steps=62434
OUT_NAME="${TASK}-${required_steps}_steps_train.png"
mkdir -p $OUTPUT_DIR

#TASK="simple_test_base_metavr_bpe_cosine-nonzero_dataset_v2-lr_1e-6_200e"
CHECKPOINT_DIR="${BASE_DIR}/${TASK}/checkpoint-${required_steps}"


GENALM_HOME="/home/jovyan/shares/SR003.nfs2/caduseus_artem/phageml/ModernBERT/downstream_tasks/phageml/regressor" MODERNBERT_HOME="/home/jovyan/shares/SR003.nfs2/caduseus_artem/phageml/ModernBERT" CUDA_VISIBLE_DEVICES=6 python downstream_tasks/phageml/regressor/inference.py \
  --config $CONFIG \
  --checkpoint $CHECKPOINT_DIR \
  --out "${OUTPUT_DIR}/${OUT_NAME}" \
  --device cuda \
  --task $TASK \
  --epoch $required_steps
