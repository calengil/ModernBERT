import re
from typing import List, Tuple, Dict

import numpy as np
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


def collect_lib1_lib2_pairs(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Находит пары колонок вида:
      lib1_X <-> lib2_X

    Собирает все пары значений по всем строкам и всем таким колонкам.
    Если в одной из колонок значение отсутствует, пара пропускается.

    Returns:
        x_lib1: np.ndarray
        y_lib2: np.ndarray
        matched_columns: список найденных пар колонок [(lib1_col, lib2_col), ...]
    """
    lib1_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("lib1_")]
    lib2_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("lib2_")]

    lib1_map: Dict[str, str] = {c[len("lib1_"):]: c for c in lib1_cols}
    lib2_map: Dict[str, str] = {c[len("lib2_"):]: c for c in lib2_cols}

    common_suffixes = sorted(set(lib1_map.keys()) & set(lib2_map.keys()))
    matched_columns = [(lib1_map[suffix], lib2_map[suffix]) for suffix in common_suffixes]

    x_vals: List[float] = []
    y_vals: List[float] = []

    for lib1_col, lib2_col in matched_columns:
        pair_df = df[[lib1_col, lib2_col]].copy()
        pair_df[lib1_col] = pd.to_numeric(pair_df[lib1_col], errors="coerce")
        pair_df[lib2_col] = pd.to_numeric(pair_df[lib2_col], errors="coerce")
        pair_df = pair_df.dropna(subset=[lib1_col, lib2_col])

        if len(pair_df) == 0:
            continue

        x_vals.extend(pair_df[lib1_col].tolist())
        y_vals.extend(pair_df[lib2_col].tolist())

    return np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float), matched_columns


def plot_lib1_lib2_scatter(
    x_lib1: np.ndarray,
    y_lib2: np.ndarray,
    save_path: str,
    figsize=(8, 8),
    zero_placeholder=1e-7,
):
    """
    Строит scatter plot и считает корреляции:
    - Pearson
    - Spearman

    Нулевые значения отображаются на log-шкале через zero_placeholder.
    """
    if len(x_lib1) == 0:
        raise ValueError("Нет валидных пар значений lib1/lib2 для построения графика.")

    pearson_r = float(np.corrcoef(x_lib1, y_lib2)[0, 1])

    rank_x = pd.Series(x_lib1).rank(method="average").to_numpy()
    rank_y = pd.Series(y_lib2).rank(method="average").to_numpy()
    spearman_r = float(np.corrcoef(rank_x, rank_y)[0, 1])

    # Для отображения на log scale:
    # все нули заменяем на маленькое положительное значение
    x_plot = np.where(x_lib1 <= 0, zero_placeholder, x_lib1)
    y_plot = np.where(y_lib2 <= 0, zero_placeholder, y_lib2)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        x_plot,
        y_plot,
        s=18,
        alpha=0.6,
        edgecolors="none",
    )

    ax.set_xscale('log')
    ax.set_yscale('log')

    vmin = min(np.min(x_plot), np.min(y_plot))
    vmax = max(np.max(x_plot), np.max(y_plot))
    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1.5)

    # Добавим отдельную подпись для "нулевого" уровня
    xticks = [zero_placeholder, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2 ]
    yticks = [zero_placeholder, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels(["0", "1e-3", "1e-2", "1e-1", "1", "1e1", "1e2"])
    ax.set_yticklabels(["0", "1e-3", "1e-2", "1e-1", "1", "1e1", "1e2"])

    ax.set_xlabel("lib1 values")
    ax.set_ylabel("lib2 values")
    ax.set_title(
        f"lib1 vs lib2\nPearson r = {pearson_r:.4f}, N = {len(x_lib1)}" #, Spearman ρ = {spearman_r:.4f}
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return pearson_r, spearman_r


# ====================== НАСТРОЙКИ ======================
biodata = "/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT/phage/biodata"
mutations_path = f"{biodata}/mutations_effect.tsv"

# Можно оставить фильтр пустым, если хотите вообще все строки
#df_filter = {}
df_filter = {"effect": ["neg", "no_effect", "pos_or_neutral"], "single_mut": [0, 1]}
# Важно: first_library_col должен указывать на первую колонку lib1_
first_library_col = "lib1_DH10_pooled_F"

save_path = f"{biodata}/lib1_vs_lib2_scatter.png"
# ======================================================

df = load_and_filter_mutation_csv(
    csv_path=mutations_path,
    first_library_col=first_library_col,
    metadata_keep=df_filter,
    sep="\t",
)

x_lib1, y_lib2, matched_columns = collect_lib1_lib2_pairs(df)

print("Найденные пары колонок:")
for c1, c2 in matched_columns:
    print(f"  {c1} <-> {c2}")

print(f"\nВсего валидных пар значений: {len(x_lib1)}")

pearson_r, spearman_r = plot_lib1_lib2_scatter(
    x_lib1=x_lib1,
    y_lib2=y_lib2,
    save_path=save_path,
)

print(f"Pearson r:  {pearson_r:.6f}")
print(f"Spearman ρ: {spearman_r:.6f}")
