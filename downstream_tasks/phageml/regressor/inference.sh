#!/bin/bash

PYTHONNOUSERSITE=1 
export PYTHONPATH=/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT:$PYTHONPATH

CKPT="base_metavr_bpe"
TASK="simple_test_${CKPT}_cosine_correct_transform"
CONFIG="/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT/finetuning/simple_test.yaml"

GENALM_HOME="/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT/finetuning" MODERNBERT_HOME="/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT" CUDA_VISIBLE_DEVICES=6 python finetuning/inference.py \
  --config $CONFIG \
  --checkpoint /home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT/finetuning/output/simple_test_base_metavr_bpe_cosine_correct_transform_continue/checkpoint-52300 \
  --out /home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT/finetuning/output/test/bpe_correct_transform_50e.png \
  --device cuda