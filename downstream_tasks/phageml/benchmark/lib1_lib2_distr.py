import re
from typing import Any, Dict, List, Optional, Tuple, Union

from Bio.Data import CodonTable
import numpy as np
import pandas as pd
import pysam
import pickle
import torch
import math
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt


y_min = 1e-2
y_max = 1e2
alpha = 1.0
zero_placeholder = 10e-5


def _transform_y(y: float, y_min=1e-2, y_max=1e2, alpha=1.0) -> torch.Tensor:
    y = min(max(y, y_min), y_max)
    logy = math.log10(y)

    z = alpha * logy
    return torch.sigmoid(torch.tensor(z, dtype=torch.float32))


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


def plot_lib_distributions_2x2(
    df: pd.DataFrame,
    lib1_col: str,
    lib2_col: str,
    save_path: str,
    bins: int = 60,
    zero_placeholder: float = 10e-5,
    figsize=(12, 10),
):
    lib1_raw = pd.to_numeric(df[lib1_col], errors="coerce").dropna().to_numpy(dtype=float)
    lib2_raw = pd.to_numeric(df[lib2_col], errors="coerce").dropna().to_numpy(dtype=float)

    lib1_transformed = np.array(
        [float(_transform_y(v).item()) for v in lib1_raw],
        dtype=float
    )
    lib2_transformed = np.array(
        [float(_transform_y(v).item()) for v in lib2_raw],
        dtype=float
    )

    lib1_raw_plot = np.where(lib1_raw <= 0, zero_placeholder, lib1_raw)
    lib2_raw_plot = np.where(lib2_raw <= 0, zero_placeholder, lib2_raw)

    lib1_transformed_plot = np.where(lib1_transformed <= 0, zero_placeholder, lib1_transformed)
    lib2_transformed_plot = np.where(lib2_transformed <= 0, zero_placeholder, lib2_transformed)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    def _make_log_hist(ax, values, title):
        vmin = max(np.min(values), zero_placeholder)
        vmax = np.max(values)

        if vmax <= vmin:
            vmax = vmin * 10

        bins_edges = np.logspace(np.log10(vmin), np.log10(vmax), bins)

        ax.hist(values, bins=bins_edges, alpha=0.85)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    def _make_transformed_hist(ax, values, title):
        bins_edges = np.linspace(0.0, 1.0, bins)

        ax.hist(values, bins=bins_edges, alpha=0.85)
        ax.set_xlim(0.0, 1.0)
        ax.set_yscale('log')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    _make_log_hist(axes[0, 0], lib1_raw_plot, f'{lib1_col} raw')
    _make_log_hist(axes[0, 1], lib2_raw_plot, f'{lib2_col} raw')
    _make_transformed_hist(axes[1, 0], lib1_transformed_plot, f'{lib1_col} transformed')
    _make_transformed_hist(axes[1, 1], lib2_transformed_plot, f'{lib2_col} transformed')

    axes[0, 0].set_ylabel('Count')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_xlabel('Value')
    axes[1, 1].set_xlabel('Value')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ====================== НАСТРОЙКИ ======================
biodata = '/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT/phage/biodata'
mutations_path = f'{biodata}/mutations_effect.tsv'

lib1_col = 'lib1_DH10_pooled_F'
lib2_col = 'lib2_DH10_pooled_F'

save_path = f'{biodata}/dh10_distributions_2x2.png'

df_filter = {}
# ======================================================

df = load_and_filter_mutation_csv(
    csv_path=mutations_path,
    first_library_col='lib1_DH10_pooled_F',
    metadata_keep=df_filter,
    sep='\t'
)

plot_lib_distributions_2x2(
    df=df,
    lib1_col=lib1_col,
    lib2_col=lib2_col,
    save_path=save_path,
    bins=60,
    zero_placeholder=zero_placeholder,
)