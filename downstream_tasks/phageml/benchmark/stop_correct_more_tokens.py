#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pysam
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import torch
from torch import nn
from transformers.utils.logging import warning_once
from scipy.stats import entropy
from tqdm import tqdm
import argparse
import datetime
import configparser

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import defaultdict, Counter
import pickle
import json
from pathlib import Path
import sys


repo_path = Path("/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT")
sys.path.insert(0, str(repo_path))

#================ MODEL =================

def build_modergena_model(checkpoint_filepath, modernbert_distr_path, name, random_init):
    from omegaconf import DictConfig, OmegaConf
    from omegaconf import OmegaConf as om
    #print (modernbert_distr_path)
    #sys.path.append(modernbert_distr_path)
    from src import flex_bert as flex_bert_module
    from src import hf_bert as hf_bert_module
    from src import mosaic_bert as mosaic_bert_module
    from src.bert_layers.model import init_mlm_model_from_pretrained
    from src.bert_layers.configuration_bert import FlexBertConfig
    from composer.utils.checkpoint import _ensure_valid_checkpoint

    def build_model(cfg: DictConfig):
        if cfg.name == "hf_bert":
            return hf_bert_module.create_hf_bert_mlm(
                pretrained_model_name=cfg.pretrained_model_name,
                use_pretrained=cfg.get("use_pretrained", None),
                model_config=cfg.get("model_config", None),
                tokenizer_name=cfg.get("tokenizer_name", None),
                gradient_checkpointing=cfg.get("gradient_checkpointing", None),
            )
        elif cfg.name == "mosaic_bert":
            return mosaic_bert_module.create_mosaic_bert_mlm(
                pretrained_model_name=cfg.pretrained_model_name,
                pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
                model_config=cfg.get("model_config", None),
                tokenizer_name=cfg.get("tokenizer_name", None),
                gradient_checkpointing=cfg.get("gradient_checkpointing", None),
            )
        elif cfg.name == "flex_bert":
            return flex_bert_module.create_flex_bert_mlm(
                pretrained_model_name=cfg.pretrained_model_name,
                pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
                model_config=cfg.get("model_config", None),
                tokenizer_name=cfg.get("tokenizer_name", None),
                gradient_checkpointing=cfg.get("gradient_checkpointing", None),
                recompute_metric_loss=cfg.get("recompute_metric_loss", False),
                disable_train_metrics=cfg.get("disable_train_metrics", False),
            )
        else:
            raise ValueError(f"Not sure how to build model with name={cfg.name}")

    cpt_dir = os.path.dirname(checkpoint_filepath)
    cfg_path = os.path.join(cpt_dir, f"{name}.txt")
    yaml_cfg = om.load(cfg_path)
    model = build_model(yaml_cfg.model)
	
    if not random_init: 
        print (f"Loading checkpoint from {checkpoint_filepath}")

        checkpoint_filepath = Path(checkpoint_filepath)
        assert checkpoint_filepath.exists(), f"Checkpoint {checkpoint_filepath} does not exist"

        # added weights_only=False to suppress this error: 
        # In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. 
        # Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code 
        # execution. Do it only if you got the file from a trusted source.

        state = torch.load(_ensure_valid_checkpoint(checkpoint_filepath), map_location="cpu", weights_only=False) 
        state_dict = state.get("state", {})
        model_state = state_dict.get("model", {})
        assert len(model_state) > 0, "Model state is empty, please check the checkpoint and checkpoint path"
        model.load_state_dict(model_state)

    return model


# Load model and tokenizer
def load_model_and_tokenizer(cpt_path, tokenizer_path, modernbert_distr_path=None, name="cfg.yaml", random_init=False):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    vocab_size = len(tokenizer)
    #print(vocab_size)

    #print(tokenizer.convert_ids_to_tokens([vocab_size-1]))
    #print(tokenizer.convert_ids_to_tokens(128))
    #raise
    model = build_modergena_model(cpt_path, modernbert_distr_path, name, random_init)
	
    if torch.cuda.is_available():
        model = model.cuda()
        model.eval()
	
    return model, tokenizer


#================ METRICS =================

from Bio.Seq import Seq

def translate_protein(seq, strand='+', frame=0, protein_direction='strand'):
    seq = Seq(seq[frame:])

    if strand == '-':
        seq = seq.reverse_complement()

    protein = str(seq.translate())

    if strand == '-' and protein_direction != 'NC':
        protein = protein[::-1]
        
    return protein

def check_stop(new_sequences, seq):

    new_proteins =[]
    
    protein = translate_protein(seq, protein_direction='NC')

    for new_seq in new_sequences:
        if '-' in new_seq:
            print(new_seq)
            raise
        new_protein = translate_protein(new_seq, protein_direction='NC')

        new_stops = new_protein.count('*')

        if protein.count('*') == 1 and protein[-1] == '*':
            if new_protein[-1] == '*':
                new_stops -= 1
        elif protein.count('*') >= 1:
            raise ValueError("ref protein has multiple or non-terminal STOP")
        
        #if '*' in new_protein:
        #    print(new_protein) 
        #with open('/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT/output/stop_bench_kmer.faa', 'a') as file:
        #    file.write(f">new_seq\n{new_protein}\n")

        new_proteins.append(new_protein)

    new_stops

    return new_stops, new_proteins

#================ SAMPLING =================
def temperature_scaled_softmax(logits, temperature: float):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return torch.softmax(logits / float(temperature), dim=-1)


def build_window_around_gene(
    fasta,                  # pysam.FastaFile
    tokenizer,
    chrom: str,
    gene_start: int,         # 0-based
    gene_end: int,           # 0-based half-open
    target_len: int = 1024,  # включая CLS/SEP
    flank_bp: int = 200_000,
    left_multiple: int | None = 6,
):
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizer must have cls_token_id and sep_token_id")

    max_payload = target_len - 2
    if max_payload <= 0:
        raise ValueError("target_len must be >= 2")

    chrom_len = fasta.get_reference_length(chrom)

    left_start = max(0, gene_start - flank_bp)
    right_end = min(chrom_len, gene_end + flank_bp)

    # --- ONE fetch
    full_seq = fasta.fetch(chrom, left_start, right_end).upper()

    # offsets inside full_seq
    gene_start_in_full = gene_start - left_start
    gene_end_in_full = gene_end - left_start

    left_seq = full_seq[:gene_start_in_full]
    gene_seq = full_seq[gene_start_in_full:gene_end_in_full]
    #print(f"gene_seq[:3] = {gene_seq[:3]}")
    #print(len(gene_seq))
    right_seq = full_seq[gene_end_in_full:]

    # --- left_multiple: trim LEFT flank from the LEFT to make it multiple
    if left_multiple is not None and left_multiple > 1 and len(left_seq) > 0:
        cut = len(left_seq) % int(left_multiple)
        if cut != 0:
            left_seq = left_seq[cut:]  # trim from the left

    # --- tokenize 3 parts once
    left_tok = tokenizer(left_seq, add_special_tokens=False)["input_ids"] if left_seq else []
    gene_tok = tokenizer(gene_seq, add_special_tokens=False)["input_ids"]
    right_tok = tokenizer(right_seq, add_special_tokens=False)["input_ids"] if right_seq else []
    #print(f"gene_tok = {len(gene_tok)}")
    # --- if gene too long: take from LEFT to RIGHT as much as fits
    if len(gene_tok) > max_payload:
        gene_tok = gene_tok[:max_payload]

    remaining = max_payload - len(gene_tok)

    # --- take flanks: equal left/right, then fill remainder from whichever side has more
    left_take = min(len(left_tok), remaining // 2)
    right_take = min(len(right_tok), remaining // 2)

    used = left_take + right_take
    leftover = remaining - used

    if leftover > 0:
        extra_left = min(len(left_tok) - left_take, leftover)
        left_take += extra_left
        leftover -= extra_left

    if leftover > 0:
        extra_right = min(len(right_tok) - right_take, leftover)
        right_take += extra_right
        leftover -= extra_right

    left_part = left_tok[-left_take:] if left_take > 0 else []
    right_part = right_tok[:right_take] if right_take > 0 else []

    payload = left_part + gene_tok + right_part

    input_ids = [cls_id] + payload + [sep_id]
    #attention_mask = [1] * len(input_ids)

    gene_start_pos = 1 + len(left_part)
    gene_end_pos = gene_start_pos + len(gene_tok)
    inner_positions = np.arange(gene_start_pos, gene_end_pos, dtype=int)
    print(f"len(input_ids) = {len(input_ids)}")
    #print(tokenizer.convert_ids_to_tokens(input_ids[inner_positions[0] - 1]))
    return input_ids, inner_positions



def random_order_sampling(model,
                          tokenizer,
                          input_ids,
                          inner_positions,
                          metrics,
                          model_name='kmer',
                          order_mode='random',
                          temperature=1.0,
                          seed=42):

    rng = np.random.default_rng(seed)
    #results = []

    #cls_id = tokenizer.cls_token_id
    #sep_id = tokenizer.sep_token_id

    assert min(inner_positions) == inner_positions[0]
    assert max(inner_positions) == inner_positions[-1]

    seq_arr = []

    token_distr = []
    token_lens_distr =[]
    same_token = 0

    f"Run seq"
    #tokenized = tokenizer(seq.upper(), add_special_tokens=False)
    #print(tokenized)

    #ids = tokenized["input_ids"]
    #attn = tokenized.get("attention_mask", None)
    #print(tokenized["input_ids"])
    token_distr_wt = input_ids[inner_positions[0]:inner_positions[-1] + 1]
    #print(f"token len = {len(token_distr_wt)}")
    #print(f"first_tok = {tokenizer.convert_ids_to_tokens(input_ids[inner_positions[0]])}")
    #print(f"last_tok = {tokenizer.convert_ids_to_tokens(input_ids[inner_positions[-1]])}")
    #ids = [cls_id] + ids + [sep_id]
    #if attn is not None:
    #    attn = [1] + attn + [1]
    #print(ids, attn)
    attn = [1] * len(input_ids)
    #input_ids = torch.tensor([ids], dtype=torch.long)
    #L = input_ids.shape[1]

    #inner_positions = np.arange(1, L - 1)
    if order_mode == 'random':
        order = rng.permutation(inner_positions)
    elif order_mode == 'normal':
        order = inner_positions

    for pos in order:

        inputs = {"input_ids": torch.tensor([input_ids], dtype=torch.long)}
        if attn is not None:
            inputs["attention_mask"] = torch.tensor([attn], dtype=torch.long)


        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}


        #print(inputs["input_ids"])

        old_id = inputs["input_ids"][0, pos].item()#.clone()
        #print(old_id)
        inputs["input_ids"][0, pos] = tokenizer.mask_token_id
        #print(inputs["input_ids"])
        #print(old_id)
        with torch.no_grad():
            outputs = model(inputs)
            logits = outputs.logits.reshape(
                inputs["input_ids"].shape[0],
                inputs["input_ids"].shape[1],
                -1
            )

        probs = temperature_scaled_softmax(logits[0, pos], temperature)

        #print(probs)
        if model_name == 'kmer':
        #    probs[:128] = 0.0
            for bad_id in [tokenizer.cls_token_id, tokenizer.sep_token_id,
                        tokenizer.pad_token_id, tokenizer.mask_token_id, 0, 5]:
                probs[bad_id] = 0.0

            s = probs.sum()
            if s <= 0:
                raise ValueError("All probabilities were masked out; check vocab/id ranges.")
            probs = probs / s

        new_id = torch.argmax(probs).item()
        #print(old_id)#.item())#.detach().cpu().tolist())
        inputs["input_ids"][0, pos] = new_id
        #rint(new_id)
        #print('---------------')
        #raise
        if old_id == new_id:
            same_token += 1
        #print(inputs["input_ids"])
        
        token_distr.append(new_id)

        old_id_len = len(tokenizer.convert_ids_to_tokens(old_id))
        new_id_len = len(tokenizer.convert_ids_to_tokens(new_id))

        token_lens_distr.append(new_id_len - old_id_len)


        out_ids = inputs["input_ids"][0].detach().cpu().tolist()[min(inner_positions):max(inner_positions) + 1]#[1:-1]
        #print(out_ids)
        #raise
        tokens = tokenizer.convert_ids_to_tokens(out_ids)
        new_seq = "".join(tokens)

        seq_arr.append(new_seq)

    metrics["same_token"] = [same_token]
    metrics["token_lens_distr"] = token_lens_distr
    metrics["token_distr"] = token_distr
    metrics["token_distr_wt"] = token_distr_wt
        #print(metrics["token_distr_wt"])
        #results.append(seq_arr)
    #raise    
    return seq_arr


def visualize_distribution(numbers, save_path):
    numbers = np.asarray(numbers)  # для удобства

    # Определяем диапазон и создаём бины строго по целым числам
    min_val = int(numbers.min())
    max_val = int(numbers.max())

    # Бины с шагом 1 (центрируем каждый бар точно на целом числе)
    bins = np.arange(min_val - 0.5, max_val + 1.5, 1)

    plt.figure(figsize=(11, 6))

    plt.hist(numbers, bins=bins, edgecolor='black', alpha=0.85)

    plt.title('len(new_id) - len(old_id) distribution', fontsize=14)
    plt.xlabel('len(new_id) - len(old_id)', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    plt.grid(True, alpha=0.3)

    # Принудительно делаем обе оси целочисленными
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Дополнительно можно ограничить шаг по X = 1 (особенно если разница маленькая)
    if max_val - min_val <= 30:           # для небольших диапазонов
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_token_distribution(numbers, save_path):
    """
    Visualizes the distribution of token frequencies:
    - Computes frequency of each unique token.
    - Plots histogram where X is the frequency (count of occurrences),
      Y is the number of unique tokens with that frequency.
    Ensures integer values on both axes.
    """
    numbers = np.asarray(numbers)  # Convert to array for convenience
    
    # Compute frequencies of each unique token
    token_freqs = Counter(numbers)
    freq_values = list(token_freqs.values())  # List of frequencies (counts)
    
    if not freq_values:
        print("No data to visualize.")
        return
    
    # Determine min and max for bins
    min_val = int(min(freq_values))
    max_val = int(max(freq_values))
    
    # Decide on bins: if range is small, use integer-centered bins; else, use auto (e.g., 50 bins)
    if max_val - min_val < 100:
        bins = np.arange(min_val - 0.5, max_val + 1.5, 1)  # Center bars on integers
    else:
        bins = 50  # Or 'auto', 'sturges', etc. for larger ranges
    
    plt.figure(figsize=(11, 6))
    
    plt.hist(freq_values, bins=bins, edgecolor='black', alpha=0.85)
    
    plt.title('Distribution of token frequencies', fontsize=14)
    plt.xlabel('Count of occurrences', fontsize=12)
    plt.ylabel('Count of unique tokens', fontsize=12)
    
    plt.grid(True, alpha=0.3)
    
    # Force both axes to have integer ticks
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # For small ranges, ensure step of 1 on X
    if max_val - min_val <= 30:
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_token_ids(token_ids, tokenizer, save_path, top_n=10):

    id_counts = Counter(token_ids)
    
    unique_ids = list(id_counts.keys())
    tokens = tokenizer.convert_ids_to_tokens(unique_ids)
    
    tokens_dict = {token: id_counts[id_] for id_, token in zip(unique_ids, tokens)}
    ids_dict = {id_: count for id_, count in id_counts.items()}
    
    analysis_dict = {
        'tokens': tokens_dict,
        'token_ids': ids_dict
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(analysis_dict, f)
    
    json_path = save_path.rsplit('.', 1)[0] + '.json' if '.' in save_path else save_path + '.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_dict, f, ensure_ascii=False, indent=4)
    print(f"Словарь сохранен как JSON в {json_path}")



    sorted_tokens = sorted(tokens_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print("Most inserted tokens:")
    for token, count in sorted_tokens:
        print(f"{token}: {count}")

def main(args):
    arr = []
    for i in tqdm(range(20)):
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(cpt_path=args.cpt_path, 
                                                    tokenizer_path=args.tokenizer_path, 
                                                    modernbert_distr_path=args.modernbert_distr_path,
                                                    name=args.name,
                                                    random_init=True)

        # Process genome and save results
        output_path = os.path.join(args.out_dir, args.name)
        fasta = pysam.Fastafile(args.fasta)

        chrom = 'V01146.1'
        start = 34624-1
        end = 36285
        sequence = fasta.fetch(reference=chrom, start=start, end=end).upper()
        #print(len(sequences[0]))
        #print(tokenizer.cls_token_id)
        #raise
        input_ids, inner_positions = build_window_around_gene(
                                                            fasta,                  # pysam.FastaFile
                                                            tokenizer,
                                                            chrom=chrom,
                                                            gene_start=start,         # 0-based
                                                            gene_end=end,           # 0-based half-open
                                                            target_len=1024,  # включая CLS/SEP
                                                            flank_bp = 100000,
                                                            left_multiple = 6, #None|6
                                                        )

        metrics = defaultdict(list)
        #metrics['same_token'] = []

        #sequences = ['ATGCTAGTAGTCGAG',
        #             'GTCAGGGACTATGTG',]
        #arr = []
        #for i in tqdm(range(10)):
        new_sequences = random_order_sampling(model,
                            tokenizer,
                            input_ids,
                            inner_positions,
                            metrics,
                            model_name=str(args.name).split('_')[-1],
                            order_mode='random',
                            temperature=1.0,
                            seed=42)
        
        #assert len(new_sequences) == len(sequences)
        #print(new_sequences[0])
        results, new_proteins = check_stop(new_sequences, sequence)
        arr.append(results)
    #print(new_proteins[0])
    print(arr)
    print(np.mean(arr))
    print(np.median(arr))    
    print(f"STOPs = {results}")
    print(f"Same_token = {metrics['same_token']}")
    print(f"Count of unique k-mers = {len(set(metrics['token_distr']))}")
    #print(metrics)
    raise
    #    arr.append(results[0])
    save_path = f"{args.out_dir}/{args.name}_token_lens_distr_more_context.png"
    #save_path = f"{args.out_dir}/{args.name}_random_init_token_lens_distr.png"
    #visualize_distribution(metrics['token_lens_distr'], save_path)
    print(len(metrics["token_distr"]))
    save_path = f"{args.out_dir}/{args.name}_token_distr_more_context.png"
    visualize_token_distribution(metrics["token_distr"], save_path)


    save_path = f"{args.out_dir}/{args.name}_token_distr_more_context.pickle"
    #save_path = f"{args.out_dir}/{args.name}_random_init_token_distr.pickle"
    analyze_token_ids(metrics["token_distr"], tokenizer, save_path, top_n=10)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=str, 
                        help="fasta file", default=None)
    parser.add_argument("--name", type=str, 
                        help="label of the experiment", default=None)
    parser.add_argument("--cpt_path", type=str, 
                        help="path to the checkpoint", default=None)
    parser.add_argument("--input_len_tokens", type=int, 
                        help="maximum input length in number of tokens", default=None)
    parser.add_argument("--tokenizer_path", type=str, 
                        help="path to the tokenizer", default=None)
    parser.add_argument("--out_dir", type=str, default="data/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--modernbert_distr_path", type=str, help="Path to ModernBERT distribution", 
                        default=os.path.expanduser("~/DNALM/")
                        )
    parser.add_argument("--seq_chunk_len",type=int, default=50_000, help="Chunk size used to split original fasta record for tokenization")
    parser.add_argument("--config", type=str, help="Path to configuration file", default=None)

    args = parser.parse_args()
        
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)   