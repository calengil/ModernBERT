import re
import pickle
import math

import numpy as np
import pandas as pd
import pysam
import torch
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

_MUT_RE = re.compile(r"^([A-Za-z\*])(\d+)([A-Za-z\*])$")


def load_and_prepare_mutation_csv(
    csv_path: str,
    library_cols: list[str],
    metadata_keep: dict | None = None,
    sep: str = ',',
) -> pd.DataFrame:
    """
    Загружает таблицу один раз и нормирует КАЖДУЮ библиотеку на её WT.

    Что делает:
    1. читает таблицу
    2. для каждой библиотеки из library_cols:
       - находит WT
       - делит всю колонку на WT этой библиотеки
    3. удаляет строку WT
    4. при необходимости применяет общие metadata-фильтры

    Важно:
    - здесь НЕТ фильтра score > 0
    - это делается позже отдельно для каждой библиотеки,
      потому что теперь library selection зависит от split и приоритета библиотеки
    """
    if metadata_keep is None:
        metadata_keep = {}

    df = pd.read_csv(csv_path, sep=sep)
    #df = df[['Mutation', 'single_mut', 'lib1_DH10_pool2_F', 'lib2_DH10_pool2_F']].copy().reset_index(drop=True)
    if "Mutation" not in df.columns:
        raise ValueError("Column 'Mutation' not found")

    for col in library_cols:
        if col not in df.columns:
            raise ValueError(f"Library column not found: {col}")

    wt_mask = df["Mutation"] == "WT"
    if wt_mask.sum() == 0:
        raise ValueError("WT row not found in mutation table")

    # Нормируем каждую библиотеку на свой WT
    for col in library_cols:
        wt_value = df.loc[wt_mask, col].values[0]
        df[col] = df[col] / wt_value

    # Удаляем WT после нормировки
    df = df[~wt_mask].copy()

    # Общие metadata-фильтры
    mask = pd.Series(True, index=df.index)
    for col, allowed_values in metadata_keep.items():
        if col not in df.columns:
            raise ValueError(f"Metadata column not found: {col}")
        if allowed_values is not None:
            mask &= df[col].isin(allowed_values)

    df = df[mask].reset_index(drop=True)
    return df


def select_split_with_library_priority(
    df: pd.DataFrame,
    split_single_mut_value: int,
    preferred_library_col: str,
    fallback_library_col: str,
    mutation_col: str = "Mutation",
) -> pd.DataFrame:
    """
    Собирает один split с приоритетом одной библиотеки.

    Логика:
    - берём строки нужного split по single_mut
    - сначала берём все Mutation, где preferred_library_col > 0
    - потом из fallback_library_col добираем только те Mutation,
      которых нет среди уже взятых preferred-мутантов

    Возвращает dataframe, где:
    - каждая строка = один sample
    - есть колонка selected_score
    - есть колонка source_library
    """
    if "single_mut" not in df.columns:
        raise ValueError("Column 'single_mut' not found")

    if mutation_col not in df.columns:
        raise ValueError(f"Column '{mutation_col}' not found")

    if preferred_library_col not in df.columns:
        raise ValueError(f"Preferred library column not found: {preferred_library_col}")

    if fallback_library_col not in df.columns:
        raise ValueError(f"Fallback library column not found: {fallback_library_col}")

    split_df = df[df["single_mut"] == split_single_mut_value].copy()

    preferred_ok = split_df[preferred_library_col].notna() & (split_df[preferred_library_col] > 0)
    fallback_ok = split_df[fallback_library_col].notna() & (split_df[fallback_library_col] > 0)

    preferred_df = split_df[preferred_ok].copy()
    preferred_df["selected_score"] = preferred_df[preferred_library_col]
    preferred_df["source_library"] = preferred_library_col

    preferred_mutations = set(preferred_df[mutation_col].astype(str))

    fallback_df = split_df[fallback_ok].copy()
    fallback_df = fallback_df[~fallback_df[mutation_col].astype(str).isin(preferred_mutations)].copy()
    fallback_df["selected_score"] = fallback_df[fallback_library_col]
    fallback_df["source_library"] = fallback_library_col

    out_df = pd.concat([preferred_df, fallback_df], axis=0, ignore_index=True)
    return out_df.reset_index(drop=True)


def make_most_frequent_codon_mutants(
    aa2codons_freq: dict[str, list[tuple[float, str]]],
    dna_seq: str,
    cds_start_0based: int,
    df: pd.DataFrame,
    fn_col: str,
    mutation_col: str = "Mutation",
) -> tuple[list[str], list[float]]:
    """
    Для каждой строки df создаёт один sample:
      - если Mutation = A12G, меняется один кодон
      - если Mutation = A12G_L45P_R100K, меняются несколько кодонов

    Для каждого aa_to выбирается самый частый кодон из aa2codons_freq.

    Возвращает:
      mut_seqs: список мутантных ДНК-последовательностей
      fn_scores: список соответствующих score

    ----------------------------
    Как работает функция по шагам
    ----------------------------

    1. Берёт одну строку таблицы.
       Эта строка уже соответствует одному sample.

    2. Читает mutation string.
       Примеры:
         A12G
         A12G_L45P_R100K

    3. Разбивает строку по символу '_':
         A12G               -> ["A12G"]
         A12G_L45P_R100K    -> ["A12G", "L45P", "R100K"]

    4. Для каждой части парсит:
         - позицию аминокислоты
         - конечную аминокислоту aa_to

       Для A12G:
         wt = A
         pos = 12
         aa_to = G

       Внутри этой функции реально используется:
         - pos
         - aa_to

    5. Для aa_to ищет самый частый кодон в aa2codons_freq.
       Например, если:
         aa2codons_freq["K"] = [(0.7, "AAA"), (0.3, "AAG")]
       то будет выбран "AAA".

    6. Переводит аминокислотную позицию в координату кодона в ДНК:
         nt0 = cds_start_0based + (pos - 1) * 3

    7. В WT-последовательности заменяет нужный кодон на best_codon.

       Для single mutation это делается один раз.
       Для multi mutation замены вносятся последовательно в одну и ту же new_dna.

    8. После обработки всех частей linked mutation
       готовая последовательность добавляется в mut_seqs,
       а score из fn_col добавляется в fn_scores.

    Итого:
      одна строка таблицы = один sample
    """
    if mutation_col not in df.columns:
        raise ValueError(f"Column '{mutation_col}' not found")
    if fn_col not in df.columns:
        raise ValueError(f"Column '{fn_col}' not found")

    dna = dna_seq.upper()
    L = len(dna)

    mut_seqs: list[str] = []
    fn_scores: list[float] = []

    for _, row in df.iterrows():
        mut_raw = row.get(mutation_col)
        if pd.isna(mut_raw):
            continue

        v = row.get(fn_col, None)
        if v is None or pd.isna(v):
            continue
        v = float(v)

        mut = str(mut_raw).strip()
        parts = mut.split("_")

        parsed_parts: list[tuple[int, str]] = []
        for part in parts:
            m = _MUT_RE.match(part.strip())
            if m is None:
                raise ValueError(f"Bad mutation format: {part} (full mutation={mut})")

            pos = int(m.group(2))
            aa_to = m.group(3).upper()
            parsed_parts.append((pos, aa_to))

        if len(parsed_parts) == 0:
            continue

        new_dna = dna

        for pos, aa_to in parsed_parts:
            if aa_to not in aa2codons_freq or not aa2codons_freq[aa_to]:
                raise ValueError(f"No codon frequencies for AA '{aa_to}' (mutation={mut})")

            _, best_codon = max(aa2codons_freq[aa_to], key=lambda x: x[0])
            best_codon = best_codon.upper()

            nt0 = cds_start_0based + (pos - 1) * 3
            if nt0 < 0 or nt0 + 3 > L:
                raise ValueError(
                    f"AA pos {pos} out of range for dna length={L} with cds_start={cds_start_0based}"
                )

            new_dna = new_dna[:nt0] + best_codon + new_dna[nt0 + 3:]

        mut_seqs.append(new_dna)
        fn_scores.append(v)

    return mut_seqs, fn_scores


def _transform_y(y: float, y_min=1e-2, y_max=1e2, alpha=1.0) -> torch.Tensor:
    y = min(max(y, y_min), y_max)
    logy = math.log10(y)
    z = alpha * logy
    return torch.sigmoid(torch.tensor(z, dtype=torch.float32))


def plot_distribution(data, save_path):
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=20, range=(0, 1), edgecolor='black', alpha=0.7)
    plt.xlabel('Fn-score transformed')
    plt.ylabel('Freq')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.ylim([0, 10000])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


biodata = '/home/jovyan/shares/SR003.nfs2/caduseus_artem/phageml/ModernBERT/downstream_tasks/phageml/biodata'
fasta_path = f'{biodata}/sequence.fasta'
mutations_path = f'{biodata}/mutations_effect_2.tsv'

lib1_col = 'lib1_DH10_pool2_F'
lib2_col = 'lib2_DH10_pool2_F'

# Общие фильтры, одинаковые для обеих библиотек
# Пример:
# df_filter = {"effect": ["neg", "no_effect", "pos_or_neutral"]}
df_filter = {}

mutation_df = load_and_prepare_mutation_csv(
    mutations_path,
    library_cols=[lib1_col, lib2_col],
    metadata_keep=df_filter,
    sep='\t'
)

# train: single_mut == 1, приоритет lib1, fallback lib2
train_df = select_split_with_library_priority(
    mutation_df,
    split_single_mut_value=1,
    preferred_library_col=lib1_col,
    fallback_library_col=lib2_col,
    mutation_col="Mutation",
)

# test: single_mut == 0, приоритет lib2, fallback lib1
test_df = select_split_with_library_priority(
    mutation_df,
    split_single_mut_value=0,
    preferred_library_col=lib2_col,
    fallback_library_col=lib1_col,
    mutation_col="Mutation",
)

print(f"train samples = {len(train_df)}")
print(train_df["source_library"].value_counts(dropna=False).to_dict())

print(f"test samples = {len(test_df)}")
print(test_df["source_library"].value_counts(dropna=False).to_dict())
#raise
dict_path = f"{biodata}/rbp_codon_usage.pkl"
with open(dict_path, 'rb') as f:
    aa2codons_freq = pickle.load(f)

fasta = pysam.Fastafile(fasta_path)

chrom = 'V01146.1'
start = 34624 - 1
end = 36285
cds_start = 0
sequence = fasta.fetch(reference=chrom, start=start, end=end).upper()

train_mut_seqs, train_fn_scores = make_most_frequent_codon_mutants(
    aa2codons_freq=aa2codons_freq,
    dna_seq=sequence,
    cds_start_0based=cds_start,
    df=train_df,
    fn_col="selected_score",
    mutation_col="Mutation",
)

test_mut_seqs, test_fn_scores = make_most_frequent_codon_mutants(
    aa2codons_freq=aa2codons_freq,
    dna_seq=sequence,
    cds_start_0based=cds_start,
    df=test_df,
    fn_col="selected_score",
    mutation_col="Mutation",
)

y_min = 1e-2
y_max = 1e2
alpha = 1.0

train_fn_scores = [_transform_y(i, y_min=y_min, y_max=y_max, alpha=alpha) for i in train_fn_scores]
test_fn_scores = [_transform_y(i, y_min=y_min, y_max=y_max, alpha=alpha) for i in test_fn_scores]

print(len(train_fn_scores))
print(len(test_fn_scores))

plot_distribution(
     train_fn_scores,
     '/home/jovyan/shares/SR003.nfs2/caduseus_artem/phageml/ModernBERT/downstream_tasks/phageml/datasets/multitest/plots/train_v2_positive_log.png'
)
plot_distribution(
     test_fn_scores,
     '/home/jovyan/shares/SR003.nfs2/caduseus_artem/phageml/ModernBERT/downstream_tasks/phageml/datasets/multitest/plots/test_v2_positive_log.png'
)
raise
output_path = '/home/jovyan/shares/SR003.nfs2/caduseus_artem/phageml/ModernBERT/downstream_tasks/phageml/datasets/multitest'

train_output = f'{output_path}/train_v2_positive.hdf5'
with h5py.File(train_output, "a") as file:
    for i, fn in tqdm(enumerate(train_fn_scores), total=len(train_fn_scores), desc='Processing train'):
        group = file.create_group(f"sample_{i}")
        group.attrs['seq'] = train_mut_seqs[i]
        group.create_dataset("fn_transform", data=np.array([fn], dtype=np.float64))

test_output = f'{output_path}/test_v2_positive.hdf5'
with h5py.File(test_output, "a") as file:
    for i, fn in tqdm(enumerate(test_fn_scores), total=len(test_fn_scores), desc='Processing test'):
        group = file.create_group(f"sample_{i}")
        group.attrs['seq'] = test_mut_seqs[i]
        group.create_dataset("fn_transform", data=np.array([fn], dtype=np.float64))