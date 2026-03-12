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

#================ MUTATIONS =================
import re
import pandas as pd
import matplotlib.pyplot as plt

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



_MUT_RE = re.compile(r"^([A-Za-z\*])(\d+)([A-Za-z\*])$")

def unique_mutation_sites(df, mutation_col: str | None = None):
    """
    Возвращает список списков позиций, уникальных по набору позиций.
    Пример:
      A102Y, A102W, A131D_G133D  ->  [[102], [131, 133]]
    """
    if mutation_col is None:
        mutation_col = df.columns[0]

    seen = set()
    out = []

    for mut in df[mutation_col].dropna().astype(str).tolist():

        m = _MUT_RE.match(mut.strip())
        if m is None:
            raise ValueError(f"Bad mutation format: {mut}")

        if m not in out:
            out.append(m)

    return out


def build_site_to_position_aa_dict(df, site_list, mutation_col: str | None = None):
    """
    Теперь:
      - df содержит только ОДНОаминокислотные замены вида "A102C" (без '_')
      - site_list = список позиций (int), например [2, 131, 133]

    Возвращает словарь:
      {
        2:   ['C','I'],   # WT, затем все MUT
        131: ['A','D'],
        ...
      }
    """
    if mutation_col is None:
        mutation_col = df.columns[0]

    site_set = set(int(x) for x in site_list)

    # промежуточно храним WT и множество MUT
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

        # WT должен быть консистентным
        if cell["wt"] is None:
            cell["wt"] = wt
        elif cell["wt"] != wt:
            raise ValueError(f"WT conflict at position {pos}: {cell['wt']} vs {wt} (mutation={mut})")

        cell["muts"].add(aa)

    # финализация: [WT] + sorted(MUT без WT)
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



#ok
def codon_first_nt_positions(aa_pos, dna_seq: str, cds_start_0based: int):
    """
    aa_positions: список аминокислотных позиций (1-based)
    dna_seq: строка ДНК
    cds_start_0based: координата первого нуклеотида CDS в dna_seq (0-based)

    Возвращает список координат (0-based) первых нуклеотидов кодонов.
    """

    L = len(dna_seq)

    # aa_pos 1-based -> сдвиг (aa_pos-1)*3
    nt0 = cds_start_0based + (int(aa_pos) - 1) * 3

    # минимальная проверка границ (кодон = 3 буквы)
    if nt0 < 0 or nt0 + 3 > L:
        raise ValueError(f"AA pos {aa_pos} out of range for dna length={L} with cds_start={cds_start_0based}")

    return nt0


#########################################################


import numpy as np
import torch
import re
from Bio.Seq import Seq


#ok
def translate_cds_to_protein(dna_seq: str, cds_start_0based: int = 0) -> str: #!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    Транслируем CDS: начинаем перевод с dna_seq[cds_start_0based:].
    strand/frame тут не трогаю, потому что в задаче cds_start уже "реальная позиция начала CDS".
    """
    cds = Seq(dna_seq[cds_start_0based:].upper())
    return str(cds.translate())



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

def find_token_index_covering_nt(offsets, nt_pos: int) -> int:
    """Индекс токена, который содержит nt_pos."""
    for i, (s, e) in enumerate(offsets):
        if s <= nt_pos < e:
            return i
    raise ValueError(f"nt_pos={nt_pos} not covered by token offsets")

def temperature_scaled_softmax(logits, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return torch.softmax(logits / float(temperature), dim=-1)

def mask_token_and_get_probs(model, base_inputs: dict, mask_pos: int, mask_token_id: int, temperature: float) -> np.ndarray:
    """
    base_inputs: {"input_ids": (1,L), "attention_mask": (1,L) (опционально)} уже на CUDA (если нужно)
    mask_pos: позиция в input_ids (включая CLS/SEP)
    Возвращает probs по всем токенам: np.ndarray shape (V,)
    """
    inputs = {k: v.clone() for k, v in base_inputs.items()}
    inputs["input_ids"][0, mask_pos] = mask_token_id

    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits.reshape(
            inputs["input_ids"].shape[0],
            inputs["input_ids"].shape[1],
            -1
        )

    probs = temperature_scaled_softmax(logits[0, mask_pos], temperature)
    return probs.detach().cpu().numpy()


def mask_regions_and_get_probs(
    model,
    tokenizer,
    dna_seq: str,
    mask_span: tuple[int, int],   # (start0, end0) 0-based, half-open
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Одна маска:
      - режем dna_seq на left + [MASK] + tail
      - токенизируем left и tail отдельно
      - склеиваем input_ids, добавляем CLS/SEP, attention_mask
      - прогоняем модель
      - возвращаем probs для [MASK] только по token_id 128..end (shape (V-128,))
    """
    if mask_span is None or len(mask_span) != 2:
        raise ValueError("mask_span must be a tuple (start, end)")

    s, e = int(mask_span[0]), int(mask_span[1])
    if not (0 <= s < e <= len(dna_seq)):
        raise ValueError(f"Bad span {(s, e)} for dna length {len(dna_seq)}")

    # --- режем ДНК и токенизируем куски
    ids = []
    attn = []

    left = dna_seq[:s]
    if len(left) > 0:
        t = tokenizer(left.upper(), add_special_tokens=False)
        ids.extend(t["input_ids"])
        attn.extend(t.get("attention_mask", [1] * len(t["input_ids"])))

    # --- одна маска = один токен
    ids.append(tokenizer.mask_token_id)
    attn.append(1)

    tail = dna_seq[e:]
    if len(tail) > 0:
        t = tokenizer(tail.upper(), add_special_tokens=False)
        ids.extend(t["input_ids"])
        attn.extend(t.get("attention_mask", [1] * len(t["input_ids"])))

    # --- добавляем CLS/SEP
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizer must have cls_token_id and sep_token_id")

    ids = [cls_id] + ids + [sep_id]
    attn = [1] + attn + [1]

    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.tensor([attn], dtype=torch.long)

    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    # --- CUDA (как у тебя)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # --- forward
    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits.reshape(
            inputs["input_ids"].shape[0],
            inputs["input_ids"].shape[1],
            -1
        )

    # --- позиция MASK (должна быть одна)
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero()
    if mask_positions.shape[0] != 1:
        raise ValueError(f"Expected exactly 1 [MASK], got {mask_positions.shape[0]}")

    pos = int(mask_positions[0, 1].item())

    # --- probs только по token_id 128..end
    vocab_size = logits.shape[-1]
    start_id = 128
    if start_id >= vocab_size:
        raise ValueError(f"Tokenizer/model vocab too small: vocab_size={vocab_size}, start_id={start_id}")

    p = temperature_scaled_softmax(logits[0, pos], temperature)  # (V,)
    return p[start_id:].detach().cpu().numpy()  # (V-128,)

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
    tokenizer_list = [tokenizer.convert_ids_to_tokens(n).upper for n in range(128, len(tokenizer) - 1)] #["GCT", "GAT", "TGG", "AAA", 'AAG']

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

        group_probs = mask_regions_and_get_probs(
            model,
            tokenizer,
            dna_seq,
            mask_region)

        group_substitution_probs = aa_probabilities_from_token_probs(
                                        probs_list=group_probs,
                                        tokens=tokenizer_list,
                                        dna_seq=dna_seq,
                                        nt_pos=nt_pos,
                                        site_to_pos_aas=substitution_dict, 
                                        site_key=aa_pos,
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
    plt.title(f"{library_col}: F-score vs prob (Pearson r={corr:.3f})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

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

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    #return {"pos": labels, "p_mut_sum": xs, "p_wt": ys}
