#!/bin/bash

DATA_DIR="/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT"

#    --name="base_metavr_kmer" \
#    --cpt_path="${DATA_DIR}/phage/ckpt/base_metavr_kmer/base_metavr_kmer_ep61-ba176100-rank0.zip" \
#    --tokenizer_path="${DATA_DIR}/data/tokenizers/6mer" \


#    --name="base_metavr_bpe" \
#    --cpt_path="${DATA_DIR}/phage/ckpt/base_metavr_bpe/base_metavr_bpe_ep44-ba115000-rank0.zip" \
#    --tokenizer_path="AIRI-Institute/gena-lm-bert-base-t2t" \

#conda activate bert24 for ModernGENA/GENA
CUDA_VISIBLE_DEVICES=1 python ./phage/codon_mutation_token.py \
    --name="base_metavr_bpe_continue" \
    --cpt_path="${DATA_DIR}/phage/ckpt/base_metavr_bpe_continue/base_metavr_bpe_ep74-ba190600-rank0.zip" \
    --tokenizer_path="AIRI-Institute/gena-lm-bert-base-t2t" \
    --out_dir="${DATA_DIR}/output/continue" \
    --modernbert_distr_path="${DATA_DIR}" \
    --fasta="${DATA_DIR}/phage/biodata/sequence.fasta" \
    --mutations_path="${DATA_DIR}/phage/biodata/mutations_effect.tsv"
