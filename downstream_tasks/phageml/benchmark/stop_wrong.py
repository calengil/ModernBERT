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

from pathlib import Path
import sys


repo_path = Path("/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT")
sys.path.insert(0, str(repo_path))

#================ MODEL =================

def build_modergena_model(checkpoint_filepath, modernbert_distr_path, name):
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
def load_model_and_tokenizer(cpt_path, tokenizer_path, modernbert_distr_path=None, name="cfg.yaml"):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    vocab_size = len(tokenizer)
    #print(vocab_size)

    #print(tokenizer.convert_ids_to_tokens([vocab_size-1]))
    #print(tokenizer.convert_ids_to_tokens(128))
    #raise
    model = build_modergena_model(cpt_path, modernbert_distr_path, name)
	
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
        new_seq = new_sequences[seq_idx]
        seq = sequences[seq_idx]
        print(seq)
        new_protein = translate_protein(new_seq, protein_direction='NC')
        protein = translate_protein(seq, protein_direction='NC')
        print(protein)
        raise
        new_stops = new_protein.count('*')

        if protein.count('*') == 1 and protein[-1] == '*':
             if new_protein[-1] == '*':
                new_stops -= 1
        elif protein.count('*') >= 1:
            raise ValueError("ref protein has multiple or non-terminal STOP")


        stop_count.append(new_stops)
        new_proteins.append(new_protein)

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
                          order_mode='random',
                          model_name='kmer',
                          temperature=1.0,
                          seed=42):

    rng = np.random.default_rng(seed)
    results = []

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    for seq in sequences:
        f"Run seq {sequences.index(seq)}"
        tokenized = tokenizer(seq.upper(), add_special_tokens=False)
        #print(tokenized)

        ids = tokenized["input_ids"]
        attn = tokenized.get("attention_mask", None)


        ids = [cls_id] + ids + [sep_id]
        if attn is not None:
            attn = [1] + attn + [1]
        #print(ids, attn)
        input_ids = torch.tensor([ids], dtype=torch.long)
        L = input_ids.shape[1]

        inputs = {"input_ids": input_ids}
        if attn is not None:
            inputs["attention_mask"] = torch.tensor([attn], dtype=torch.long)


        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}


        inner_positions = np.arange(1, L - 1)
        if order_mode == 'random':
            order = rng.permutation(inner_positions)
        elif order_mode == 'normal':
            order = inner_positions
        ###print(f"order = {order}")
        ###id_test = 0
        #for pos in tqdm(order):
        for pos in order:
            #print(f"pos = {pos}")
            ###if id_test < 2:
            ###    print(pos)
            ###    print(inputs["input_ids"][0, pos])
            inputs["input_ids"][0, pos] = tokenizer.mask_token_id
            #print(inputs["input_ids"][0, pos])
            #print(inputs)
            #new_id = 0 ### test ckpt
            #while new_id < 128 or new_id > 4096: ### test ckpt     
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
            #new_id = torch.argmax(logits[0, pos]).item()
            #new_id = torch.multinomial(probs, num_samples=1).item()
            #print(new_id)
            #print("--------------")

            inputs["input_ids"][0, pos] = new_id
            ###if id_test < 2:
                ###print(inputs["input_ids"][0, pos])
                ###print("----------------------------")
            ###id_test += 1
            
                
        out_ids = inputs["input_ids"][0].detach().cpu().tolist()[1:-1]
        #print(f"out = {out_ids}")
        #raise
        tokens = tokenizer.convert_ids_to_tokens(out_ids)
        #for i in out_ids:
        #    print(tokenizer.convert_ids_to_tokens([i]))

        results.append("".join(tokens))
        
    return results



def main(args):

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(cpt_path=args.cpt_path, 
                                                tokenizer_path=args.tokenizer_path, 
                                                modernbert_distr_path=args.modernbert_distr_path,
                                                name=args.name)

    # Process genome and save results
    output_path = os.path.join(args.out_dir, args.name)
    fasta = pysam.Fastafile(args.fasta)

    chrom = 'V01146.1'
    start = 34624-1
    end = 36285
    sequences = [fasta.fetch(reference=chrom, start=start, end=end).upper()]

    metrics = {}
    metrics['same_token'] = 0


    #sequences = ['ATGCTAGTAGTCGAG',
    #             'GTCAGGGACTATGTG',]
    #arr = []
    #for i in tqdm(range(10)):
    new_sequences = random_order_sampling(model,
                        tokenizer,
                        sequences,
                        metrics,
                        order_mode='random',
                        model_name='kmer',
                        temperature=1.0,
                        seed=42)
    
    print(new_sequences[0])
    results, new_proteins = check_stop(new_sequences, sequences)
    print(new_proteins[0])
    print(results[0])
    #    arr.append(results[0])


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