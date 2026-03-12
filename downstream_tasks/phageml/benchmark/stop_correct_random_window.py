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

import random
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

def check_stop(new_sequences, sequences):
    stop_count = []
    new_proteins =[]
    
    for seq_idx in range(len(new_sequences)):
        new_seq_arr = new_sequences[seq_idx]
        seq = sequences[seq_idx]
        #print(seq)
        protein = translate_protein(seq, protein_direction='NC')
        #print(protein)
        #raise
        new_proteins_arr = []
        for new_seq in new_seq_arr:
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

            new_proteins_arr.append(new_protein)

        stop_count.append(new_stops)
        new_proteins.append(new_proteins_arr)

    return stop_count, new_proteins

#================ SAMPLING =================
def temperature_scaled_softmax(logits, temperature: float):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return torch.softmax(logits / float(temperature), dim=-1)


def random_order_sampling(model,
                          tokenizer,
                          sequences,
                          metrics,
                          model_name='kmer',
                          order_mode='random',
                          temperature=1.0,
                          seed=42):

    rng = np.random.default_rng(seed)
    results = []

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    for seq in tqdm(sequences):
        seq_arr = []

        token_distr = []
        token_lens_distr =[]
        same_token = 0

        f"Run seq {sequences.index(seq)}"
        tokenized = tokenizer(seq.upper(), add_special_tokens=False)
        #print(tokenized)

        ids = tokenized["input_ids"]
        attn = tokenized.get("attention_mask", None)
        #print(tokenized["input_ids"])
        token_distr_wt = tokenized["input_ids"]

        ids = [cls_id] + ids + [sep_id]
        if attn is not None:
            attn = [1] + attn + [1]
        #print(ids, attn)
        input_ids = torch.tensor([ids], dtype=torch.long)
        L = input_ids.shape[1]

        inner_positions = np.arange(1, L - 1)
        if order_mode == 'random':
            order = rng.permutation(inner_positions)
        elif order_mode == 'normal':
            order = inner_positions

        for pos in order:

            inputs = {"input_ids": input_ids}
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
                            tokenizer.pad_token_id, tokenizer.mask_token_id]:
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

            token_lens_distr.append((new_id_len, old_id_len))


            out_ids = inputs["input_ids"][0].detach().cpu().tolist()[1:-1]
            #print(out_ids)
            #raise
            tokens = tokenizer.convert_ids_to_tokens(out_ids)
            new_seq = "".join(tokens)

            seq_arr.append(new_seq)

        metrics["same_token"].append(same_token)
        metrics["token_lens_distr"].append(token_lens_distr)
        metrics["token_distr"].append(token_distr)
        metrics["token_distr_wt"].append(token_distr_wt)
        #print(metrics["token_distr_wt"])
        results.append(seq_arr)
        
    return results


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


def visualize_distribution_token_lens(lens_arr, save_path):
    counts = Counter(lens_arr)
    unique_points = list(counts.keys())
    x_coords = [p[0] for p in unique_points]
    y_coords = [p[1] for p in unique_points]
    sizes = [counts[p] * 30 for p in unique_points]  # Множитель 10 можно скорректировать по вкусу

    plt.figure(figsize=(8, 6))
    plt.scatter(x_coords, y_coords, s=sizes, alpha=0.7)
    plt.xlabel('New token lens')
    plt.ylabel('Old token lens')
    #plt.title('Scatter plot из кортежей')
    plt.grid(True, alpha=0.3)


    for i in range(len(unique_points)):
        x = x_coords[i]
        y = y_coords[i]
        count = counts[unique_points[i]]
        plt.text(x, y, str(count), fontsize=14, ha='center', va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_token_ids(token_ids, tokenizer, save_path, top_n=10):

    #print(token_ids[:10])
    id_counts = Counter(token_ids)
    
    unique_ids = list(id_counts.keys())
    #print(unique_ids[:10])
    #raise
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


def random_windows(seq_length, window_size, num_windows=1, seed=None, non_overlapping=True):
    if window_size > seq_length:
        raise ValueError("Window size cannot be larger than sequence length")

    if seed is not None:
        random.seed(seed)

    if not non_overlapping:
        max_start = seq_length - window_size
        windows = []
        for _ in range(num_windows):
            start = random.randint(0, max_start)
            end = start + window_size
            windows.append((start, end))
        return sorted(windows)

    # For non_overlapping=True
    # Check if possible
    adjusted_max = seq_length - num_windows * window_size + num_windows
    if adjusted_max < num_windows:
        raise ValueError("Cannot place all windows without overlapping due to insufficient space")

    # Sample transformed positions
    ts = random.sample(range(adjusted_max), num_windows)
    ts.sort()
    starts = [ts[i] + i * (window_size - 1) for i in range(num_windows)]
    windows = [(start, start + window_size) for start in starts]

    return windows

def visualize_token_distribution_random(numbers, save_path, mode=None):
    """
    Visualizes the distribution of token frequencies:
    - If mode is None, concatenates all sub-arrays into one and computes frequencies as before.
    - If mode is 'median' or 'mean', computes frequencies per sub-array, then aggregates per token using median or mean (with 0 for missing).
    - Plots histogram where X is the (aggregated) frequency,
      Y is the number of unique tokens with that frequency.
    Ensures appropriate ticks on axes.
    """
    if mode not in (None, "median", "mean"):
        raise ValueError("mode must be None, 'median' or 'mean'")

    if mode is None:
        # Flatten all sub-arrays
        flat = []
        for sub in numbers:
            flat.extend(sub)
        token_freqs = Counter(flat)
        freq_values = list(token_freqs.values())
    else:
        # Collect all unique tokens
        all_tokens = set()
        counters = []
        for sub in numbers:
            c = Counter(sub)
            counters.append(c)
            all_tokens.update(c.keys())

        # Aggregate
        aggregated = {}
        for token in all_tokens:
            freqs = [c.get(token, 0) for c in counters]
            if mode == "mean":
                agg = float(np.mean(freqs))
            elif mode == "median":
                agg = float(np.median(freqs))
            aggregated[token] = agg

        freq_values = list(aggregated.values())

    if not freq_values:
        print("No data to visualize.")
        return

    # Check if all frequencies are integers (or integer floats)
    are_integers = all(isinstance(v, int) or (isinstance(v, float) and v.is_integer()) for v in freq_values)
    if are_integers:
        freq_values = [int(v) for v in freq_values]

    # Determine min and max
    min_val = min(freq_values)
    max_val = max(freq_values)

    # Decide on bins
    if are_integers:
        if max_val - min_val < 100:
            bins = np.arange(min_val - 0.5, max_val + 1.5, 1)  # Center bars on integers
        else:
            bins = 50
    else:
        bins = 50  # Or 'auto' for floats

    plt.figure(figsize=(11, 6))
    
    n, bins, patches = plt.hist(freq_values, bins=bins, edgecolor='black', alpha=0.85)
    
    # Добавляем числа над барами
    for i in range(len(patches)):
        if n[i] > 0:  # Только если значение > 0
            plt.text(patches[i].get_x() + patches[i].get_width() / 2, n[i], int(n[i]), ha='center', va='bottom', fontsize=10)
    
    plt.title('Distribution of token frequencies', fontsize=14)
    plt.xlabel('Count of occurrences', fontsize=12)
    plt.ylabel('Count of unique tokens', fontsize=12)
    
    plt.grid(True, alpha=0.3)
    
    # Force axes ticks
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Y is always int
    
    if are_integers:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if max_val - min_val <= 30:
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main(args):
    arr = []
    metrics = defaultdict(list)
    #for i in tqdm(range(20)):
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(cpt_path=args.cpt_path, 
                                                tokenizer_path=args.tokenizer_path, 
                                                modernbert_distr_path=args.modernbert_distr_path,
                                                name=args.name,
                                                random_init=False)

    # Process genome and save results
    output_path = os.path.join(args.out_dir, args.name)
    fasta = pysam.Fastafile(args.fasta)

    chrom = 'V01146.1'
    start = 34624-1
    end = 36285
    window_len = end - start
    n_seq = 20
    print('start!!!!!!!!!!!!!')
    random_windows_arr = random_windows(fasta.get_reference_length(chrom), window_len, num_windows=n_seq, seed=42)
    #print(random_windows_arr)
    
    sequences = [fasta.fetch(reference=chrom, start=start, end=end).upper() for (start, end) in random_windows_arr]

    
    #metrics['same_token'] = []

    #sequences = ['ATGCTAGTAGTCGAG',
    #             'GTCAGGGACTATGTG',]

    new_sequences = random_order_sampling(model,
                        tokenizer,
                        sequences,
                        metrics,
                        model_name=str(args.name).split('_')[-1],
                        order_mode='random',
                        temperature=1.0,
                        seed=42)
    
    assert len(new_sequences) == len(sequences)
    #print(metrics["token_distr"])
#    metrics["token_distr"].append(token_distr)
#    metrics["token_distr_wt"].append(token_distr_wt)
    save_path = f"{args.out_dir}/{args.name}_token_distr_random_{n_seq}.png"
    #visualize_token_distribution_random(metrics["token_distr"], save_path, mode=None)

    save_path = f"{args.out_dir}/{args.name}_token_distr_wt.pickle"
    #save_path = f"{args.out_dir}/{args.name}_random_init_token_distr.pickle"
    analyze_token_ids([item for sublist in metrics["token_distr_wt"] for item in sublist], tokenizer, save_path, top_n=10)


    raise
    save_path = f"{args.out_dir}/{args.name}_token_distr_wt_random_{n_seq}.png"
    visualize_token_distribution_random(metrics["token_distr"], save_path, mode=None)
    raise

    #print(new_sequences[0])
    results, new_proteins = check_stop(new_sequences, sequences)
    #print(results[0])
    #arr.append(results[0])
    #print(new_proteins[0])
    print('-----------------')
    #print(arr)
    #print(np.mean(arr))
    #print(np.median(arr))
    print(f"STOPs = {np.sum(results)}")
    print(f"Same_token = {metrics['same_token']}")
    print(f"Count of unique tokens = {len(set(metrics['token_distr'][0]))}")

    raise
    save_path = f"{args.out_dir}/{args.name}_token_lens_distr.png"
    visualize_distribution_token_lens(metrics['token_lens_distr'][0], save_path)

    raise
    #print(metrics)
    #raise
    #    arr.append(results[0])
    save_path = f"{args.out_dir}/{args.name}_token_lens_distr.png"
    #save_path = f"{args.out_dir}/{args.name}_random_init_token_lens_distr.png"
    #visualize_distribution(metrics['token_lens_distr'][0], save_path)

    save_path = f"{args.out_dir}/{args.name}_token_distr_wt.png"
    visualize_token_distribution(metrics["token_distr_wt"][0], save_path)


    save_path = f"{args.out_dir}/{args.name}_token_distr_wt.pickle"
    #save_path = f"{args.out_dir}/{args.name}_random_init_token_distr.pickle"
    analyze_token_ids(metrics["token_distr_wt"][0], tokenizer, save_path, top_n=10)

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