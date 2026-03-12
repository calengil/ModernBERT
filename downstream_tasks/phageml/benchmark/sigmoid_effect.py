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


def plot_transformed_scores_on_sigmoid(
    ys,
    ys_transform_,
    effects,
    scores_col,
    save_path,
    color_map=None,
    alpha=1.0,
    figsize=(13, 7),
):
    if color_map is None:
        color_map = {
            "no_effect": "gray",
            "neg": "red",
            "pos_or_neutral": "limegreen"
        }

    ys = np.asarray(ys, dtype=float)
    effects = np.asarray(effects)

    transformed = np.array([
        float(v.item()) if isinstance(v, torch.Tensor) else float(v)
        for v in ys_transform_
    ], dtype=float)

    ys_clipped = np.clip(ys, y_min, y_max)
    zs = alpha * np.log10(ys_clipped)

    fig, ax = plt.subplots(figsize=figsize)

    # Фиксированные пределы осей
    x_min = alpha * np.log10(y_min)
    x_max = alpha * np.log10(y_max)

    # Кривая сигмоиды
    x_curve = np.linspace(x_min, x_max, 600)
    y_curve = 1 / (1 + np.exp(-x_curve))

    # Сначала рисуем линию
    ax.plot(
        x_curve,
        y_curve,
        linewidth=3,
        zorder=1,
    )

    # Потом точки поверх линии
    for eff, color in color_map.items():
        mask = (effects == eff)
        if np.any(mask):
            ax.scatter(
                zs[mask],
                transformed[mask],
                c=color,
                s=60,
                edgecolors='black',
                linewidth=0.5,
                alpha=0.85,
                label=f'{eff} ({np.sum(mask)} samples)',
                zorder=3,
            )

    ax.legend(title='Effect', title_fontsize=11, fontsize=10)
    ax.grid(True, alpha=0.3)

    # Верхняя ось: исходные значения y
    def z_to_y(z_val):
        return 10 ** (z_val / alpha)

    def y_to_z(y_val):
        y_val = np.asarray(y_val)
        y_val = np.clip(y_val, 1e-12, None)
        return alpha * np.log10(y_val)

    secax = ax.secondary_xaxis('top', functions=(z_to_y, y_to_z))

    # Убираем верхние деления и подписи, чтобы не было "20, 40, 60, 80, 100"
    secax.set_xticks([])
    secax.set_xticklabels([])
    secax.tick_params(top=False, labeltop=False)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ====================== НАСТРОЙКИ ======================
scores_col = 'lib1_DH10_pooled_F'   # ←←← ИЗМЕНИТЕ НА НУЖНУЮ КОЛОНКУ

y_min = 1e-2
y_max = 1e2
alpha = 1.0

color_map = {
    "no_effect": "gray",
    "neg": "red",
    "pos_or_neutral": "limegreen"
}
# ======================================================

biodata = '/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT/phage/biodata'
fasta_path = f'{biodata}/sequence.fasta'
mutations_path = f'{biodata}/mutations_effect.tsv'

df_filter = {"effect": ["neg", "no_effect", "pos_or_neutral"], "single_mut": [1]}
df = load_and_filter_mutation_csv(
    csv_path=mutations_path,
    first_library_col='lib1_DH10_pooled_F',
    metadata_keep=df_filter,
    sep='\t'
)

df_plot = df.dropna(subset=[scores_col, 'effect']).copy()

ys = df_plot[scores_col].astype(float).values
effects = df_plot['effect'].astype(str).values
ys_transform_ = [_transform_y(i) for i in ys]

ys = np.asarray(ys, dtype=float)
ys_transform_ = np.asarray(ys_transform_, dtype=object)
effects = np.asarray(effects)

save_dir = '/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT/phage/biodata'

for eff in ["no_effect", "neg", "pos_or_neutral"]:
    mask = (effects == eff)

    if np.any(mask):
        save_path = f'{save_dir}/sigmoid_effect_{eff}.png'

        plot_transformed_scores_on_sigmoid(
            ys=ys[mask],
            ys_transform_=ys_transform_[mask],
            effects=effects[mask],
            scores_col=scores_col,
            save_path=save_path,
            color_map=color_map,
            alpha=alpha,
        )