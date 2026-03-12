#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import re
from typing import Any, Dict, List, Optional, Tuple, Union

from Bio.Data import CodonTable


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


def plot_score_dist(df, first_library_col, mode="all",
                    single_col=None, col_from=None, col_to=None,
                    bins=100, out_png="dist.png",
                    logx=True, logy=True, min_pos=None,
                    transform_y=False,   # <--- NEW
                    y_min=1e-2, y_max=1e2, alpha=2.0):
    first_lib_idx = first_library_col if isinstance(first_library_col, int) else df.columns.get_loc(first_library_col)

    if mode == "single":
        j = single_col if isinstance(single_col, int) else df.columns.get_loc(single_col)
        vals = df.iloc[:, j].to_numpy()
        title = f"{df.columns[j]}"
    elif mode == "range":
        j0 = col_from if isinstance(col_from, int) else df.columns.get_loc(col_from)
        j1 = col_to   if isinstance(col_to, int)   else df.columns.get_loc(col_to)
        if j0 > j1: j0, j1 = j1, j0
        vals = df.iloc[:, j0:j1+1].to_numpy().ravel()
        title = f"{df.columns[j0]} .. {df.columns[j1]}"
    elif mode == "all":
        vals = df.iloc[:, first_lib_idx:].to_numpy().ravel()
        title = f"all cols from {df.columns[first_lib_idx]}"
    else:
        raise ValueError("mode must be: single / range / all")

    # to numeric + finite
    vals = pd.to_numeric(vals, errors="coerce")
    vals = vals[np.isfinite(vals)]

    # optionally transform (clamp -> log10 -> alpha* -> sigmoid)
    if transform_y:
        v = np.clip(vals, y_min, y_max)
        v = np.log10(v)
        v = alpha * v
        vals = 1.0 / (1.0 + np.exp(-v))  # sigmoid

        # after transform everything is (0,1), log scales usually not desired
        logx = False
        logy = False
        min_pos = None

    n_zero = int(np.sum(vals == 0))
    pos = vals[vals > 0]

    if logx and pos.size == 0:
        raise ValueError("No positive values to show on log-x histogram.")

    if min_pos is None and pos.size > 0:
        min_pos = np.min(pos)

    if pos.size > 0 and min_pos is not None:
        pos = pos[pos >= min_pos]

    plt.figure()

    if logx:
        b = np.logspace(np.log10(np.min(pos)), np.log10(np.max(pos)), bins)
        plt.hist(pos, bins=b)
        plt.xscale("log")
    else:
        plt.hist(vals, bins=bins)  # if not logx, we can include zeros too

    if logy:
        plt.yscale("log")

    plt.title(f"Fn-score distribution ({title})\nN={len(vals)} | zeros={n_zero} | positives={len(pos)}"
              + (" | TRANSFORMED" if transform_y else ""))
    plt.xlabel("Fn-score (transformed)" if transform_y else "Fn-score")
    plt.ylabel("count")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

modernbert_path = '/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT/phage'
mutations_path = f'{modernbert_path}/biodata/mutations_effect.tsv'

first_library_col = 'lib1_DH10_pooled_F'
df_filter = {"effect": ["no_effect", "neg", "pos_or_neutral"], "single_mut": [1]}
#df_filter = {"effect": ["neg", "pos_or_neutral"], "single_mut": [1]}
mutation_df = load_and_filter_mutation_csv(mutations_path,
                                            metadata_keep=df_filter,
                                            first_library_col='lib1_DH10_pooled_F',
                                            sep='\t')

output=f"{modernbert_path}/biodata/single_mut_dist_{first_library_col}_transform_a1_logy.png"
plot_score_dist(mutation_df, first_library_col, mode="single",
                    single_col=first_library_col, col_from=None, col_to=None,
                    bins=100, out_png=output,
                    transform_y=True, y_min=1e-2, y_max=1e2, alpha=1.0)