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

import re
import pandas as pd
import matplotlib.pyplot as plt
import re
from Bio.Seq import Seq


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


#================ MUTATIONS =================

_MUT_RE = re.compile(r"^([A-Za-z\*])(\d+)([A-Za-z\*])$")

def load_and_filter_mutation_csv(
    csv_path: str,
    first_library_col,  # column_name (str) OR column_idx (int)
    metadata_keep: dict | None = None,  # {"effect": ["no_effect", "neg", "pos_or_neutral"], "single_mut": [1]}
    sep=',',
):

    if metadata_keep is None:
        metadata_keep = {}


    df = pd.read_csv(csv_path, sep=sep)

    if isinstance(first_library_col, int):
        first_lib_idx = first_library_col
    else:
        first_lib_idx = df.columns.get_loc(first_library_col)

    meta_cols = list(df.columns[1:first_lib_idx])

    mask = pd.Series(True, index=df.index)
    for col in meta_cols:
        if col in metadata_keep and metadata_keep[col] is not None:
            mask &= df[col].isin(metadata_keep[col])

    return df.loc[mask].copy()


def unique_mutation_sites(df, mutation_col: str | None = None):
    """
    A102Y, A102W, A131D  ->  [102, 131]
    (уникальные позиции, 1-based)
    """
    if mutation_col is None:
        mutation_col = df.columns[0]

    seen = set()
    out = []

    for mut in df[mutation_col].dropna().astype(str).tolist():
        m = _MUT_RE.match(mut.strip())
        if m is None:
            raise ValueError(f"Bad mutation format: {mut}")

        pos = int(m.group(2))
        if pos not in seen:
            seen.add(pos)
            out.append(pos)

    return out


def build_site_to_position_aa_dict(df, site_list, mutation_col: str | None = None):
    """
    return dict:
      {
        2:   ['C','I'],   # WT,  MUT
        131: ['A','D'],
        ...
      }
    """
    if mutation_col is None:
        mutation_col = df.columns[0]

    site_set = set(int(x) for x in site_list)

    tmp = {pos: {"wt": None, "muts": set()} for pos in site_set}

    for mut in df[mutation_col].dropna().astype(str).tolist():
        if "_" in mut:
            raise ValueError(f"Expected single mutation, got linked: {mut}")

        m = _MUT_RE.match(mut.strip())
        if m is None:
            raise ValueError(f"Bad mutation format: {mut}")

        wt = m.group(1).upper()
        pos = int(m.group(2))
        aa = m.group(3).upper()

        if pos not in site_set:
            continue

        cell = tmp[pos]

        if cell["wt"] is None:
            cell["wt"] = wt
        elif cell["wt"] != wt:
            raise ValueError(f"WT conflict at position {pos}: {cell['wt']} vs {wt} (mutation={mut})")

        cell["muts"].add(aa)

    out = {}
    for pos in sorted(site_set):
        wt = tmp[pos]["wt"]
        if wt is None:
            out[pos] = []
            continue
        muts_sorted = sorted(tmp[pos]["muts"])
        muts_sorted = [x for x in muts_sorted if x != wt]
        out[pos] = [wt] + muts_sorted

    return out


def codon_first_nt_positions(aa_pos, dna_seq: str, cds_start_0based: int):
    L = len(dna_seq)

    nt0 = cds_start_0based + (int(aa_pos) - 1) * 3

    if nt0 < 0 or nt0 + 3 > L:
        raise ValueError(f"AA pos {aa_pos} out of range for dna length={L} with cds_start={cds_start_0based}")

    return nt0


def translate_cds_to_protein(dna_seq: str, cds_start_0based: int = 0) -> str:
    cds = Seq(dna_seq[cds_start_0based:].upper())
    return str(cds.translate())


def temperature_scaled_softmax(logits, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return torch.softmax(logits / float(temperature), dim=-1)


def mask_regions_and_get_probs(
    model,
    tokenizer,
    dna_seq: str,
    mask_span: tuple[int, int],   # (start0, end0) 0-based, half-open
    temperature: float = 1.0,
):
    """
    Одна маска:
      - режем dna_seq на left + [MASK] + tail
      - токенизируем left и tail отдельно
      - склеиваем input_ids, добавляем CLS/SEP, attention_mask
      - прогоняем модель
      - возвращаем:
          probs_128: np.ndarray shape (V-128,)
          tokens_128: list[str] длины (V-128) в том же порядке
        где V = vocab_size модели (logits.shape[-1])
    """
    if mask_span is None or len(mask_span) != 2:
        raise ValueError("mask_span must be a tuple (start, end)")

    s, e = int(mask_span[0]), int(mask_span[1])
    if not (0 <= s < e <= len(dna_seq)):
        raise ValueError(f"Bad span {(s, e)} for dna length {len(dna_seq)}")

    ids = []
    attn = []

    left = dna_seq[:s]
    if len(left) > 0:
        t = tokenizer(left.upper(), add_special_tokens=False)
        ids.extend(t["input_ids"])
        attn.extend(t.get("attention_mask", [1] * len(t["input_ids"])))

    ids.append(tokenizer.mask_token_id)
    attn.append(1)

    tail = dna_seq[e:]
    if len(tail) > 0:
        t = tokenizer(tail.upper(), add_special_tokens=False)
        ids.extend(t["input_ids"])
        attn.extend(t.get("attention_mask", [1] * len(t["input_ids"])))

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizer must have cls_token_id and sep_token_id")

    ids = [cls_id] + ids + [sep_id]
    attn = [1] + attn + [1]

    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.tensor([attn], dtype=torch.long)
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits.reshape(
            inputs["input_ids"].shape[0],
            inputs["input_ids"].shape[1],
            -1
        )

    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero()
    if mask_positions.shape[0] != 1:
        raise ValueError(f"Expected exactly 1 [MASK], got {mask_positions.shape[0]}")
    pos = int(mask_positions[0, 1].item())

    vocab_size_model = logits.shape[-1]
    start_id = 6 #6 #128
    if start_id >= vocab_size_model:
        raise ValueError(f"Tokenizer/model vocab too small: vocab_size={vocab_size_model}, start_id={start_id}")

    p = temperature_scaled_softmax(logits[0, pos], temperature)  # (V,)
    probs_128 = p[start_id:].detach().cpu().numpy()  # (V-128,)

    # Важно: берём токены в ТОМ ЖЕ диапазоне id, что и probs_128
    token_ids = list(range(start_id, vocab_size_model))
    tokens_128 = tokenizer.convert_ids_to_tokens(token_ids)

    # защита: иногда tokenizer может вернуть меньше/больше (редко), проверим
    if len(tokens_128) != len(probs_128):
        raise ValueError(
            f"tokens/probs length mismatch: tokens={len(tokens_128)} vs probs={len(probs_128)}. "
            "Likely tokenizer vocab size != model vocab size."
        )

    return probs_128, tokens_128

def aa_probabilities_from_token_probs(
    probs: np.ndarray,            # probs для ОДНОГО маскирования (в том же порядке, что tokens)
    tokens: list[str],            # tokenizer.convert_ids_to_tokens для тех же token_id (в том же порядке)
    dna_seq: str,
    nt_pos: int,                  # первый нуклеотид кодона (0-based)
    site_to_pos_aas: dict,        # {aa_pos: [WT, mut1, mut2, ...]}
    aa_pos: int,                  # позиция аминокислоты (1-based)
) -> dict[str, list[float]]:
    """
    Для одного локуса (одна аминокислотная позиция):

      - tail = dna[nt_pos+3 : nt_pos+5] (если есть)
      - для каждого токена tok:
          s = tok + tail
          codon = первые 3 нуклеотида s
          aa = translate_codon(codon)
          если aa разрешена для aa_pos (берём из site_to_pos_aas[aa_pos]),
          то сохраняем вероятность probs[j] в out[aa].append(prob)

    Возвращает:
      {
        "A": [p1, p2, ...],
        "D": [p3, ...],
        ...
      }
    """
    if len(probs) != len(tokens):
        raise ValueError(f"Length mismatch: probs={len(probs)} vs tokens={len(tokens)}")

    if aa_pos not in site_to_pos_aas:
        raise KeyError(f"aa_pos {aa_pos} not found in site_to_pos_aas")

    dna_seq = dna_seq.upper()
    allowed = set(x.upper() for x in site_to_pos_aas[aa_pos])

    # tail = два нуклеотида после кодона (если есть)
    tail = ""
    if nt_pos + 3 < len(dna_seq):
        tail = dna_seq[nt_pos + 3 : min(nt_pos + 5, len(dna_seq))]  # длина 0/1/2

    out: dict[str, list[float]] = {}

    for tok, p in zip(tokens, probs):
        if tok is None:
            continue
        tok = tok.upper()

        # подставляем токен перед tail, берём первые 3 нуклеотида
        s = tok + tail
        if len(s) < 3:
            continue

        codon = s[:3]
        aa = translate_cds_to_protein(codon)  # должна быть определена у тебя

        if aa in allowed:
            out.setdefault(aa, []).append(float(p))

    return out



def collect_prob(model,
                tokenizer,
                df_filtered,
                site_groups,     # output unique_mutation_sites(df_filtered)
                dna_seq,
                cds_start):

    dna_seq = dna_seq.upper()
    #tokenizer_list = [tokenizer.convert_ids_to_tokens(n).upper() for n in range(10)] #["GCT", "GAT", "TGG", "AAA", 'AAG']
    #print(tokenizer_list)
    #raise
    # токенизация ДНК и базовый input для модели (1 раз)
    #ids_no_special, attn_no_special, toks_no_special, offsets = tokenize_with_offsets(tokenizer, dna_seq)

    #cls_id = tokenizer.cls_token_id
    #sep_id = tokenizer.sep_token_id

    #ids = [cls_id] + ids_no_special + [sep_id]
    #attn = [1] + attn_no_special + [1]

    #base_inputs = {"input_ids": torch.tensor([ids], dtype=torch.long)}
    #base_inputs["attention_mask"] = torch.tensor([attn], dtype=torch.long)

    #if torch.cuda.is_available():
    #    base_inputs = {k: v.cuda() for k, v in base_inputs.items()}


    substitution_dict = build_site_to_position_aa_dict(df_filtered, site_groups) #{(2, 3): {2: ["A", "D", "K"], 3: ["A", "W"],}}

    out = {}
    for aa_pos in site_groups:
        nt_pos = codon_first_nt_positions(aa_pos, dna_seq, cds_start)

        mask_region = (nt_pos, nt_pos+3)

        group_probs, tokenizer_list = mask_regions_and_get_probs(
            model,
            tokenizer,
            dna_seq,
            mask_region)

        #print(len(group_probs))
        #raise
        group_substitution_probs = aa_probabilities_from_token_probs(
                                        probs=np.array(group_probs),
                                        tokens=tokenizer_list,
                                        dna_seq=dna_seq,
                                        nt_pos=nt_pos,
                                        site_to_pos_aas=substitution_dict, 
                                        aa_pos=aa_pos,
                                    )
        out[aa_pos] = group_substitution_probs #{2: {'A': [0.55], 'D': [0.25], 'K': [0.05, 0.05]}, 3: {'A': [0.1], 'W': [0.8]}}


    return out

def parse_single_mutation(mut_str: str):
    """
    'A133D' -> ('A', 133, 'D')
    Только одноаминокислотные замены. Если встретится '_' — упадёт.
    """
    mut_str = str(mut_str).strip()
    if "_" in mut_str:
        raise ValueError(f"Expected single mutation, got linked: {mut_str}")
    m = _MUT_RE.match(mut_str)
    if m is None:
        raise ValueError(f"Bad mutation format: {mut_str}")
    return m.group(1).upper(), int(m.group(2)), m.group(3).upper()


def plot_library_corr_fscore_vs_prob(
    df,
    prob_dict: dict,            # новый формат: {pos: {AA: [probs...]}}
    library_col: str,
    save_path: str,             # куда сохранить png/pdf
    mutation_col: str | None = None,
    wt_fscore: float = 1.0,
):
    """
    Новый формат prob_dict:
      { 133: {'A':[...], 'D':[...], ...}, 131: {...}, ... }

    Для одной библиотеки строит scatter:
      x = вероятность (sum по списку в prob_dict[pos][AA])
      y = F-score (из df для мутантов, и wt_fscore для WT)
    WT точки отмечаются красным.

    Возвращает:
      (pearson_corr, points_dict)
    """
    if mutation_col is None:
        mutation_col = df.columns[0]

    # 1) оставляем только строки, где есть F-score в этой библиотеке
    df_lib = df[df[library_col].notna()].copy()

    xs, ys, is_wt, labels = [], [], [], []

    # --- точки для мутантов (из df)
    for _, row in df_lib.iterrows():
        mut_str = row[mutation_col]
        wt_aa, pos, mut_aa = parse_single_mutation(mut_str)

        if pos not in prob_dict:
            continue

        aa_to_probs = prob_dict[pos]
        p_mut = float(np.sum(aa_to_probs.get(mut_aa, [])))
        f_mut = float(row[library_col])

        xs.append(p_mut)
        ys.append(f_mut)
        is_wt.append(False)
        labels.append(mut_str)

    # --- WT точки: по уникальным позициям, которые реально есть в df_lib
    positions_in_df = sorted({parse_single_mutation(m)[1] for m in df_lib[mutation_col].astype(str).tolist()})
    for pos in positions_in_df:
        if pos not in prob_dict:
            continue

        # WT аминокислота для позиции берём из любой строки в df (она одна)
        wt_aa = None
        for m in df_lib[mutation_col].astype(str).tolist():
            w, p, _ = parse_single_mutation(m)
            if p == pos:
                wt_aa = w
                break
        if wt_aa is None:
            continue

        aa_to_probs = prob_dict[pos]
        p_wt = float(np.sum(aa_to_probs.get(wt_aa, [])))

        xs.append(p_wt)
        ys.append(float(wt_fscore))
        is_wt.append(True)
        labels.append(f"{wt_aa}{pos}{wt_aa}")

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    is_wt = np.asarray(is_wt, dtype=bool)

    # 2) корреляция Pearson
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() >= 2 and np.std(xs[mask]) > 0 and np.std(ys[mask]) > 0:
        corr = float(np.corrcoef(xs[mask], ys[mask])[0, 1])
    else:
        corr = np.nan

    # 3) plot + save
    plt.figure()
    plt.scatter(xs[~is_wt], ys[~is_wt])
    plt.scatter(xs[is_wt], ys[is_wt], color="red")
    plt.xlabel("Probability (sum of token probs)")
    plt.ylabel(f"F-score ({library_col})")
    plt.title(f"{library_col}: F-score vs prob (Pearson r={corr})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    #plt.xscale('log')
    plt.yscale('log')
    plt.xlim([0, 1])

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    points = {"x": xs, "y": ys, "is_wt": is_wt, "labels": labels}
    #return corr, points


def plot_position_mutprob_vs_wtprob(
    df,
    prob_dict: dict,            # новый формат: {pos: {AA: [probs...]}}
    library_col: str,
    save_path: str,             # куда сохранить png/pdf
    mutation_col: str | None = None,
):
    """
    По позициям строит scatter:
      x = суммарная вероятность всех замен из df в этой позиции
      y = вероятность сохранить WT в этой позиции

    Учитываем только строки df, где есть F-score в library_col.
    """
    if mutation_col is None:
        mutation_col = df.columns[0]

    df_lib = df[df[library_col].notna()].copy()

    # соберём мутанты по позиции из df
    pos_to_wt = {}
    pos_to_mutset = {}

    for mut_str in df_lib[mutation_col].dropna().astype(str).tolist():
        wt_aa, pos, mut_aa = parse_single_mutation(mut_str)
        pos_to_wt[pos] = wt_aa
        pos_to_mutset.setdefault(pos, set()).add(mut_aa)

    xs, ys, labels = [], [], []

    for pos in sorted(pos_to_mutset.keys()):
        if pos not in prob_dict:
            continue

        aa_to_probs = prob_dict[pos]
        wt_aa = pos_to_wt[pos]

        p_wt = float(np.sum(aa_to_probs.get(wt_aa, [])))

        p_mut_sum = 0.0
        for mut_aa in pos_to_mutset[pos]:
            p_mut_sum += float(np.sum(aa_to_probs.get(mut_aa, [])))

        xs.append(p_mut_sum)
        ys.append(p_wt)
        labels.append(pos)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel("Sum probability of DF mutations at position")
    plt.ylabel("Probability to keep WT at position")
    plt.title(f"{library_col}: per-position mut prob vs WT prob")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    plt.xlim([0, 1])
    plt.ylim([0, 1])


    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    #return {"pos": labels, "p_mut_sum": xs, "p_wt": ys}

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
    sequence = fasta.fetch(reference=chrom, start=start, end=end).upper()

    df_filter = {"effect": ["no_effect", "neg", "pos_or_neutral"], "single_mut": [1]}

    mutation_df = load_and_filter_mutation_csv(args.mutations_path,
                                                metadata_keep=df_filter,
                                                first_library_col='lib1_DH10_pooled_F',
                                                sep='\t')
    
    sites_list = unique_mutation_sites(mutation_df)

    mutations_probs = collect_prob(model,
                                tokenizer,
                                mutation_df,
                                sites_list,     # output unique_mutation_sites(df_filtered)
                                sequence,
                                0)
    
    lib_name = 'lib1_DH10_pooled_F'

    save_path=f"{args.out_dir}/{args.name}_{lib_name}_mutprob_vs_wtprob.png"
    plot_position_mutprob_vs_wtprob(
                                mutation_df,
                                prob_dict=mutations_probs,
                                library_col=lib_name,
                                save_path=save_path,)


    raise

    save_path=f"{args.out_dir}/{args.name}_{lib_name}_corr_fscore_vs_prob.png"
    plot_library_corr_fscore_vs_prob(
                                mutation_df,
                                prob_dict=mutations_probs,
                                library_col=lib_name,
                                save_path=save_path,)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutations_path", type=str, 
                        help="mutation file", default=None)    
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