#!/bin/bash

#cd /home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT
#conda activate bert24;
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT:$PYTHONPATH

CONFIG_DIR="/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT/finetuning/configs"
#cd /home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT
CKPT="base_metavr_bpe"
TASK="simple_test_${CKPT}_cosine_new_dataset"
CONFIG="${CONFIG_DIR}/simple_test_bpe_new_dataset.yaml"


MODERNBERT_HOME="/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT" CUDA_VISIBLE_DEVICES=6 python finetuning/train_with_accelerate.py \
    --config $CONFIG \
    --output_dir "/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT/finetuning/output/${TASK}"

TASK="simple_test_${CKPT}_cosine_dropout_0.1"
CONFIG="${CONFIG_DIR}/simple_test_bpe_dropout_0.1.yaml"


MODERNBERT_HOME="/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT" CUDA_VISIBLE_DEVICES=6 python finetuning/train_with_accelerate.py \
    --config $CONFIG \
    --output_dir "/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT/finetuning/output/${TASK}"    