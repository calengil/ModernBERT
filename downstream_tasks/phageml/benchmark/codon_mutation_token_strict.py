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


def tokenize_with_offsets(tokenizer, dna_seq: str):
    """
    Токенизируем без спец-токенов.
    offsets[i] = (start_nt, end_nt) в координатах dna_seq (0-based),
    построено по длинам token-строк (как у тебя).
    """
    tokenized = tokenizer(dna_seq.upper(), add_special_tokens=False)
    ids = tokenized["input_ids"]
    attn = tokenized["attention_mask"]

    toks = tokenizer.convert_ids_to_tokens(ids)

    offsets = []
    cur = 0
    for t in toks:
        if t is None:
            raise ValueError("Tokenizer returned None token for input sequence.")
        offsets.append((cur, cur + len(t)))
        cur += len(t)

    # в твоём примере это считается инвариантом
    if cur != len(dna_seq):
        raise ValueError(f"Token lengths sum != dna length: {cur} vs {len(dna_seq)}")

    return ids, attn, toks, offsets

def mask_regions_and_get_probs(
    model,
    tokenizer,
    ids_no_special: list[int],           # токены всей ДНК без CLS/SEP
    attn_no_special: list[int],          # attention_mask без CLS/SEP
    offsets: list[tuple[int, int]],      # offsets для ids_no_special (0-based nt spans)
    dna_seq: str,
    mask_span: tuple[int, int],          # (start0, end0) 0-based, half-open (кодон)
    tokenizer_name: str = 'kmer',
    temperature: float = 1.0,
):
    """
    Маскирует ВСЕ токены, которые перекрывают mask_span (кодон может попасть в 1 или 2 токена),
    затем ОДИН раз прогоняет модель на последовательности с несколькими [MASK].

    Возвращает:
      probs_list:  список np.ndarray (по одному на каждый замаскированный токен),
                  каждый формы (V-128,) для token_id 128..V-1
      tokens_128:  список строк (общий для всех масок) в том же порядке, что probs_128
      masked_token_positions: список позиций [MASK] в input_ids (с CLS), в порядке слева направо
    """
    if mask_span is None or len(mask_span) != 2:
        raise ValueError("mask_span must be a tuple (start, end)")

    s, e = int(mask_span[0]), int(mask_span[1])
    if not (0 <= s < e <= len(dna_seq)):
        raise ValueError(f"Bad span {(s, e)} for dna length {len(dna_seq)}")

    if len(ids_no_special) != len(attn_no_special) or len(ids_no_special) != len(offsets):
        raise ValueError("ids_no_special, attn_no_special, offsets must have the same length")

    # 1) найти индексы токенов (в ids_no_special), которые перекрывают маскируемый регион
    token_indices = []
    for i, (ts, te) in enumerate(offsets):
        if ts < e and te > s:   # overlap
            token_indices.append(i)

    if len(token_indices) == 0:
        raise ValueError(f"No tokens overlap mask_span={mask_span}")
    token_indices = sorted(token_indices)  # слева направо

    # 2) сделать копию ids_no_special и замаскировать ВСЕ эти токены
    ids_masked = list(ids_no_special)
    for i in token_indices:
        ids_masked[i] = tokenizer.mask_token_id

    # 3) собрать base_inputs как ты хочешь
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizer must have cls_token_id and sep_token_id")

    ids = [cls_id] + ids_masked + [sep_id]
    attn = [1] + list(attn_no_special) + [1]

    base_inputs = {"input_ids": torch.tensor([ids], dtype=torch.long)}
    base_inputs["attention_mask"] = torch.tensor([attn], dtype=torch.long)

    if torch.cuda.is_available():
        base_inputs = {k: v.cuda() for k, v in base_inputs.items()}

    # 4) forward один раз
    with torch.no_grad():
        outputs = model(base_inputs)
        logits = outputs.logits.reshape(
            base_inputs["input_ids"].shape[0],
            base_inputs["input_ids"].shape[1],
            -1
        )

    # 5) позиции [MASK] в input_ids (учитывая CLS => +1)
    masked_token_positions = [i + 1 for i in token_indices]

    # sanity check: что в этих местах реально MASK
    for pos in masked_token_positions:
        if int(base_inputs["input_ids"][0, pos].item()) != int(tokenizer.mask_token_id):
            raise ValueError("Internal error: expected [MASK] at masked_token_positions")

    # 6) probs и tokens для id 128..V-1
    vocab_size_model = logits.shape[-1]
    if tokenizer_name == 'kmer':
        start_id = 128
    elif tokenizer_name == 'bpe':
        start_id = 6
    else:
        raise
    
    if start_id >= vocab_size_model:
        raise ValueError(f"Tokenizer/model vocab too small: vocab_size={vocab_size_model}, start_id={start_id}")

    token_ids = list(range(start_id, vocab_size_model))
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    probs_list = []
    for pos in masked_token_positions:
        p = temperature_scaled_softmax(logits[0, pos], temperature)  # (V,)
        probs = p[start_id:].detach().cpu().numpy()              # (V-128,)
        if len(probs) != len(tokens):
            raise ValueError("tokens/probs length mismatch (model vocab vs tokenizer vocab)")
        probs_list.append(probs)

    return probs_list, tokens, token_indices # masked_token_positions

def mask_regions_and_get_probs_old(
    base_inputs,
    model,
    tokenizer,
    offsets,
    dna_seq: str,
    mask_span: tuple[int, int],   # (start0, end0) 0-based, half-open
    temperature: float = 1.0,
):

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
    start_id = 128
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

def aa_probabilities_from_token_probs_old(
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

_CODON_TABLE = {
    "TTT":"F","TTC":"F","TTA":"L","TTG":"L",
    "CTT":"L","CTC":"L","CTA":"L","CTG":"L",
    "ATT":"I","ATC":"I","ATA":"I","ATG":"M",
    "GTT":"V","GTC":"V","GTA":"V","GTG":"V",
    "TCT":"S","TCC":"S","TCA":"S","TCG":"S",
    "CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "ACT":"T","ACC":"T","ACA":"T","ACG":"T",
    "GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*",
    "CAT":"H","CAC":"H","CAA":"Q","CAG":"Q",
    "AAT":"N","AAC":"N","AAA":"K","AAG":"K",
    "GAT":"D","GAC":"D","GAA":"E","GAG":"E",
    "TGT":"C","TGC":"C","TGA":"*","TGG":"W",
    "CGT":"R","CGC":"R","CGA":"R","CGG":"R",
    "AGT":"S","AGC":"S","AGA":"R","AGG":"R",
    "GGT":"G","GGC":"G","GGA":"G","GGG":"G",
}
def translate_codon(c):
    c = c.upper()
    if len(c) != 3 or not (set(c) <= set("ACGT")):
        return "X"
    return _CODON_TABLE.get(c, "X")


def aa_probs_from_masked_token_distributions(
    probs_list: list[np.ndarray],
    tokens: list[str],
    dna_seq: str,
    offsets: list[tuple[int, int]],
    masked_token_positions: list[int],
    mask_span: tuple[int, int],
    site_to_pos_aas: dict,
    aa_pos: int,
    cds_start_0based: int = 0,   # NEW: CDS start in dna_seq (0-based)
):
    if len(probs_list) != len(masked_token_positions):
        raise ValueError("probs_list and masked_token_positions must have the same length")
    if len(probs_list) not in (1, 2):
        raise ValueError("Expected 1 or 2 masked token distributions")
    for probs in probs_list:
        if len(probs) != len(tokens):
            raise ValueError(f"Length mismatch: probs={len(probs)} vs tokens={len(tokens)}")
    if aa_pos not in site_to_pos_aas:
        raise KeyError(f"aa_pos {aa_pos} not found in site_to_pos_aas")
    if len(site_to_pos_aas[aa_pos]) == 0:
        raise ValueError(f"site_to_pos_aas[{aa_pos}] is empty; cannot determine WT")

    wt_aa = str(site_to_pos_aas[aa_pos][0]).upper()
    allowed = set(x.upper() for x in site_to_pos_aas[aa_pos])

    dna_seq = dna_seq.upper()
    s0 = int(mask_span[0])
    if s0 < 0 or s0 + 3 > len(dna_seq):
        raise ValueError("mask_span start is out of range for original dna_seq")

    if cds_start_0based < 0 or cds_start_0based >= len(dna_seq):
        raise ValueError("cds_start_0based out of range")

    # target AA index in 0-based protein coordinates
    target_aa_idx = aa_pos - 1

    def apply_token_replace(seq: str, tok_start: int, tok_end: int, tok: str) -> str:
        return seq[:tok_start] + tok + seq[tok_end:]

    def codon_aa_at(seq: str, aa_idx0: int) -> str:
        """
        AA for codon number aa_idx0 (0-based) in the CDS starting at cds_start_0based.
        """
        c0 = cds_start_0based + aa_idx0 * 3
        codon = seq[c0:c0+3]
        if len(codon) != 3:
            return "X"
        return translate_cds_to_protein(codon)

    def affected_aa_indices(tok_start: int, tok_end: int) -> range:
        """
        Which AA indices (0-based in protein) can change due to editing seq[tok_start:tok_end).
        Only indices whose codons overlap [tok_start:tok_end) in CDS coordinates.
        """
        # restrict to CDS overlap
        start = max(tok_start, cds_start_0based)
        end = min(tok_end, len(dna_seq))
        if end <= start:
            return range(0, 0)

        a0 = (start - cds_start_0based) // 3
        a1 = (end - 1 - cds_start_0based) // 3
        if a1 < 0:
            return range(0, 0)
        a0 = max(a0, 0)
        return range(a0, a1 + 1)

    # WT AA cache for touched positions
    wt_aa_cache = {}
    def wt_aa_at(idx0: int) -> str:
        if idx0 not in wt_aa_cache:
            wt_aa_cache[idx0] = codon_aa_at(dna_seq, idx0)
        return wt_aa_cache[idx0]

    def is_synonymous_outside_target(seq_new: str, affected_idxs: set[int]) -> bool:
        """
        For all affected AA indices except target_aa_idx, AA must remain WT.
        """
        for idx0 in affected_idxs:
            if idx0 == target_aa_idx:
                continue
            if codon_aa_at(seq_new, idx0) != wt_aa_at(idx0):
                return False
        return True

    # ---- 1 mask
    if len(probs_list) == 1:
        tok_idx = masked_token_positions[0]
        if tok_idx < 0 or tok_idx >= len(offsets):
            raise ValueError(f"masked token index out of range: {tok_idx}")

        tok_start, tok_end = offsets[tok_idx]
        old_len = tok_end - tok_start

        affected_idxs = set(affected_aa_indices(tok_start, tok_end))

        prefix = dna_seq[:tok_start]
        suffix = dna_seq[tok_end:]

        best = {aa: 0.0 for aa in allowed}
        probs = probs_list[0]

        for tok, p in tqdm(list(zip(tokens, probs))):
            if tok is None:
                continue
            tok = tok.upper()

            # no indels
            if len(tok) != old_len:
                continue

            new_dna = prefix + tok + suffix

            # allow changes outside target codon only if synonymous in affected codons
            if not is_synonymous_outside_target(new_dna, affected_idxs):
                continue

            codon = new_dna[s0:s0+3]
            aa = translate_cds_to_protein(codon)

            if aa in best:
                pv = float(p)
                if pv > best[aa]:
                    best[aa] = pv

        p_wt = float(best.get(wt_aa, 0.0))
        fn = {}
        for aa in allowed:
            p = float(best.get(aa, 0.0))
            if p_wt <= 0.0:
                fn[aa] = np.inf if p > 0.0 else 0.0
            else:
                fn[aa] = p / p_wt
        return {aa_pos: fn}

    # ---- 2 masks
    tok_idx1 = masked_token_positions[0]
    tok_idx2 = masked_token_positions[1]
    if tok_idx1 < 0 or tok_idx1 >= len(offsets) or tok_idx2 < 0 or tok_idx2 >= len(offsets):
        raise ValueError("masked token index out of range")

    (s1, e1) = offsets[tok_idx1]
    (s2, e2) = offsets[tok_idx2]
    p1 = probs_list[0]
    p2 = probs_list[1]

    # left-to-right
    if s2 < s1:
        (tok_idx1, tok_idx2) = (tok_idx2, tok_idx1)
        (s1, e1), (s2, e2) = (s2, e2), (s1, e1)
        p1, p2 = p2, p1

    if e1 > s2:
        raise ValueError("masked token spans overlap; expected 1 or 2 distinct tokens")

    old1 = e1 - s1
    old2 = e2 - s2

    affected1 = set(affected_aa_indices(s1, e1))
    affected2 = set(affected_aa_indices(s2, e2))
    affected_union = affected1 | affected2

    best = {aa: 0.0 for aa in allowed}

    prefix1 = dna_seq[:s1]
    middle12 = dna_seq[e1:s2]
    token2_orig = dna_seq[s2:e2]
    suffix2_orig = dna_seq[e2:]

    for tok_a, pa in tqdm(list(zip(tokens, p1))):
        if tok_a is None:
            continue
        tok_a = tok_a.upper()
        pa = float(pa)
        if pa == 0.0:
            continue

        # no indels
        if len(tok_a) != old1:
            continue

        dna_after_first = prefix1 + tok_a + middle12 + token2_orig + suffix2_orig

        # with no indels, token2 coordinates unchanged
        s2_new, e2_new = s2, e2

        if not is_synonymous_outside_target(dna_after_first, affected_union):
            continue

        prefix2 = dna_after_first[:s2_new]
        suffix2 = dna_after_first[e2_new:]

        for tok_b, pb in zip(tokens, p2):
            if tok_b is None:
                continue
            tok_b = tok_b.upper()
            pb = float(pb)
            if pb == 0.0:
                continue

            # no indels
            if len(tok_b) != old2:
                continue

            new_dna = prefix2 + tok_b + suffix2

            if not is_synonymous_outside_target(new_dna, affected_union):
                continue

            codon = new_dna[s0:s0+3]
            aa = translate_cds_to_protein(codon)

            if aa in best:
                pv = pa * pb
                if pv > best[aa]:
                    best[aa] = pv

    p_wt = float(best.get(wt_aa, 0.0))
    fn = {}
    for aa in allowed:
        p = float(best.get(aa, 0.0))
        if p_wt <= 0.0:
            fn[aa] = np.inf if p > 0.0 else 0.0
        else:
            fn[aa] = p / p_wt

    return {aa_pos: fn}

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
    ids_no_special, attn_no_special, toks_no_special, offsets = tokenize_with_offsets(tokenizer, dna_seq)


    substitution_dict = build_site_to_position_aa_dict(df_filtered, site_groups) #{(2, 3): {2: ["A", "D", "K"], 3: ["A", "W"],}}

    fn_scores = {}

    for aa_pos in tqdm(site_groups):
        nt_pos = codon_first_nt_positions(aa_pos, dna_seq, cds_start)

        mask_region = (nt_pos, nt_pos+3)

        #print('pre-probs')
        group_probs, tokenizer_list, token_indices = mask_regions_and_get_probs(
            model,
            tokenizer,
            ids_no_special,
            attn_no_special,
            offsets,
            dna_seq,
            mask_region,
            tokenizer_name='kmer')
        #print('pre-fn')
        #print(len(group_probs))
        #raise
        group_substitution_fn_scores = aa_probs_from_masked_token_distributions(
                                        probs_list=group_probs,
                                        tokens=tokenizer_list,
                                        dna_seq=dna_seq,
                                        offsets=offsets,
                                        masked_token_positions=token_indices,
                                        mask_span=mask_region,
                                        site_to_pos_aas=substitution_dict, 
                                        aa_pos=aa_pos,
                                    )
        
        #print('fn')
        #out[aa_pos] = group_substitution_probs #{2: {'A': [0.55], 'D': [0.25], 'K': [0.05, 0.05]}, 3: {'A': [0.1], 'W': [0.8]}}
        fn_scores.update(group_substitution_fn_scores)

    return fn_scores

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


def plot_corr_fnscore_vs_fscore(
    df: pd.DataFrame,
    fn_scores: dict,                  # {aa_pos: {AA: Fn-score}}
    save_path: str,
    mutation_col: str | None = None,
    first_library_col: str | int | None = None,  # где начинаются библиотеки
    libraries: list[str] | None = None,          # если задано -> только эти колонки
    fscore_agg: str = "median",                  # "median" или "mean"
    logscale_mode=False
):
    """
    Строит scatter Fn-score vs aggregated F-score для каждой замены (строки df)
    и считает Pearson корреляцию.

    Для каждой мутации (A133D):
      Fn = fn_scores[133]['D']
      F  = agg(F-scores across libraries) по этой строке

    save_path: куда сохранить картинку.
    """
    if mutation_col is None:
        mutation_col = df.columns[0]

    # Определяем колонки библиотек
    if libraries is None:
        if first_library_col is None:
            raise ValueError("Provide either libraries list or first_library_col")
        if isinstance(first_library_col, int):
            first_idx = first_library_col
        else:
            first_idx = df.columns.get_loc(first_library_col)
        lib_cols = list(df.columns[first_idx:])
    else:
        lib_cols = [c for c in libraries if c in df.columns]

    if fscore_agg not in ("median", "mean"):
        raise ValueError("fscore_agg must be 'median' or 'mean'")

    xs_fn = []
    ys_f = []
    labels = []

    for _, row in df.iterrows():
        mut_str = row.get(mutation_col, None)
        if mut_str is None or pd.isna(mut_str):
            continue

        wt, pos, mut_aa = parse_single_mutation(mut_str)

        # Fn-score из словаря
        if pos not in fn_scores:
            continue
        fn_map = fn_scores[pos]
        if mut_aa not in fn_map:
            continue

        fn_val = fn_map[mut_aa]
        if fn_val is None or not np.isfinite(fn_val):
            continue

        # Собираем F-score по библиотекам в этой строке
        vals = []
        for col in lib_cols:
            v = row.get(col, np.nan)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            try:
                vals.append(float(v))
            except Exception:
                continue

        if len(vals) == 0:
            continue

        if fscore_agg == "median":
            f_val = float(np.median(vals))
        else:
            f_val = float(np.mean(vals))

        xs_fn.append(float(fn_val))
        ys_f.append(f_val)
        labels.append(mut_str)

    xs_fn = np.asarray(xs_fn, dtype=float)
    ys_f = np.asarray(ys_f, dtype=float)

    # Pearson correlation (не округляем)
    m = np.isfinite(xs_fn) & np.isfinite(ys_f)
    if m.sum() >= 2 and np.std(xs_fn[m]) > 0 and np.std(ys_f[m]) > 0:
        corr = float(np.corrcoef(xs_fn[m], ys_f[m])[0, 1])
    else:
        corr = np.nan

    # Plot
    plt.figure()
    plt.scatter(xs_fn, ys_f)
    plt.xlabel("Fn-score = P(AA) / P(WT)")
    plt.ylabel(f"F-score ({fscore_agg} across libraries)")
    plt.title(f"Fn-score vs F-score (Pearson r={corr})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    if logscale_mode:
        plt.xscale('log')
        plt.yscale('log')


    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    return corr, {"fn": xs_fn, "f": ys_f, "labels": labels, "lib_cols": lib_cols}

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

    mutations_fn = collect_prob(model,
                                tokenizer,
                                mutation_df,
                                sites_list,     # output unique_mutation_sites(df_filtered)
                                sequence,
                                0)
    

    corr_mode = "median"
    save_path=f"{args.out_dir}/{args.name}_{corr_mode}_fnscore_vs_fscore_strict.png"


    plot_corr_fnscore_vs_fscore(
        df=mutation_df,
        fn_scores=mutations_fn,
        save_path=save_path,
        mutation_col="Mutation",
        first_library_col="lib1_DH10_pooled_F",
        fscore_agg=corr_mode,
        logscale_mode=False

    )

    corr_mode = "mean"
    save_path=f"{args.out_dir}/{args.name}_{corr_mode}_fnscore_vs_fscore_strict.png"


    plot_corr_fnscore_vs_fscore(
        df=mutation_df,
        fn_scores=mutations_fn,
        save_path=save_path,
        mutation_col="Mutation",
        first_library_col="lib1_DH10_pooled_F",
        fscore_agg=corr_mode,
        logscale_mode=False
    )





    corr_mode = "median"
    save_path=f"{args.out_dir}/{args.name}_{corr_mode}_fnscore_vs_fscore_strict_logscale.png"


    plot_corr_fnscore_vs_fscore(
        df=mutation_df,
        fn_scores=mutations_fn,
        save_path=save_path,
        mutation_col="Mutation",
        first_library_col="lib1_DH10_pooled_F",
        fscore_agg=corr_mode,
        logscale_mode=True
    )

    corr_mode = "mean"
    save_path=f"{args.out_dir}/{args.name}_{corr_mode}_fnscore_vs_fscore_strict_logscale.png"


    plot_corr_fnscore_vs_fscore(
        df=mutation_df,
        fn_scores=mutations_fn,
        save_path=save_path,
        mutation_col="Mutation",
        first_library_col="lib1_DH10_pooled_F",
        fscore_agg=corr_mode,
        logscale_mode=True
    )




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