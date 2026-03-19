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

def build_site_to_aa_to_libtuple(
    df: pd.DataFrame,
    first_library_col: Union[str, int],   # column name OR index
    mutation_col: Optional[str] = None,   # default: df.columns[0]
    wt_value: Union[float, int, List[Any], Tuple[Any, ...]] = 1.0,
    keep_position_order: str = "appearance",  # "appearance" | "sorted"
) -> Tuple[Dict[int, Dict[str, Tuple[Any, ...]]], List[str]]:
    """
    Builds:
      {
        102: {
          "A": (1, 1, 1, ...),              # WT (filled with wt_value unless present in df)
          "C": (0.60, 0.43, 0.73, None...),
          "D": (0.0,  0.0,  None, ...),
        },
        ...
      }

    Returns (site_to_aa_to_tuple, lib_cols_in_tuple_order)
    """

    if mutation_col is None:
        mutation_col = df.columns[0]

    # library columns
    if isinstance(first_library_col, int):
        first_lib_idx = first_library_col
    else:
        first_lib_idx = df.columns.get_loc(first_library_col)

    if first_lib_idx < 0 or first_lib_idx >= len(df.columns):
        raise ValueError(f"first_library_col points outside df columns: idx={first_lib_idx}")

    lib_cols = list(df.columns[first_lib_idx:])
    n_lib = len(lib_cols)
    if n_lib == 0:
        raise ValueError("No library columns detected (empty lib_cols).")

    # WT tuple template
    if isinstance(wt_value, (list, tuple, np.ndarray)):
        if len(wt_value) != n_lib:
            raise ValueError(f"wt_value has len={len(wt_value)} but expected {n_lib} (number of lib cols)")
        wt_tuple = tuple(None if pd.isna(x) else x for x in wt_value)
    else:
        wt_tuple = tuple([wt_value] * n_lib)

    # tmp storage
    tmp: Dict[int, Dict[str, Any]] = {}  # pos -> {"wt": "A", "values": {aa: tuple}}
    pos_order: List[int] = []

    for _, row in df.iterrows():
        mut_raw = row.get(mutation_col)
        if pd.isna(mut_raw):
            continue

        mut = str(mut_raw).strip()
        if not mut or mut.lower() == "nan":
            continue

        if "_" in mut:
            raise ValueError(f"Expected single mutation, got linked: {mut}")

        m = _MUT_RE.match(mut)
        if m is None:
            raise ValueError(f"Bad mutation format: {mut}")

        wt = m.group(1).upper()
        pos = int(m.group(2))
        aa = m.group(3).upper()

        if pos not in tmp:
            tmp[pos] = {"wt": wt, "values": {}}
            pos_order.append(pos)
        else:
            if tmp[pos]["wt"] != wt:
                raise ValueError(f"WT conflict at position {pos}: {tmp[pos]['wt']} vs {wt} (mutation={mut})")

        if aa in tmp[pos]["values"]:
            raise ValueError(f"Duplicate row for {wt}{pos}{aa} (aa={aa}, pos={pos})")

        # Build tuple of library values with None for missing
        vals: List[Any] = []
        for v in row[lib_cols].tolist():
            if pd.isna(v):
                vals.append(None)
            else:
                if isinstance(v, np.generic):
                    v = v.item()
                # keep numbers as float for consistency
                if isinstance(v, (int, float, np.number)):
                    vals.append(float(v))
                elif isinstance(v, str) and v.strip() == "":
                    vals.append(None)
                else:
                    vals.append(v)

        tmp[pos]["values"][aa] = tuple(vals)

    # finalize
    if keep_position_order == "sorted":
        positions = sorted(tmp.keys())
    elif keep_position_order == "appearance":
        positions = pos_order
    else:
        raise ValueError("keep_position_order must be 'appearance' or 'sorted'")

    out: Dict[int, Dict[str, Tuple[Any, ...]]] = {}
    for pos in positions:
        wt = tmp[pos]["wt"]
        values_map: Dict[str, Tuple[Any, ...]] = tmp[pos]["values"]

        aa_dict: Dict[str, Tuple[Any, ...]] = {}

        # WT always first
        aa_dict[wt] = values_map.get(wt, wt_tuple)

        # then mutants in order of appearance in df (dict preserves insertion order)
        for aa, tup in values_map.items():
            if aa == wt:
                continue
            aa_dict[aa] = tup

        if len(aa_dict) < 2:
            raise ValueError(f"Position {pos} has only WT; expected at least one mutant")

        # sanity check: tuple lengths
        for aa, tup in aa_dict.items():
            if len(tup) != n_lib:
                raise AssertionError(f"Tuple length mismatch at pos={pos}, aa={aa}: {len(tup)} != {n_lib}")

        out[pos] = aa_dict

    return out, lib_cols

def unique_mutation_sites(df, mutation_col: str | None = None):
    """
    A102Y, A102W, A131D  ->  [102, 131]
    (1-based)
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


def generate_single_codon_variants_dedup(
    dna_seq: str,
    site_to_aa: Dict[int, Dict[str, Union[List[Any], Tuple[Any, ...]]]],
    cds_start_0based: int,
    aa_positions: List[int],
    genetic_code_id: int = 1,          # 1 = Standard
    include_wt: bool = True,           # <--- NEW: whether to include WT AA for each position
) -> Tuple[List[str], List[Union[List[Any], Tuple[Any, ...]]]]:
    """
    Generate DNA sequences with exactly ONE codon replaced for each position in aa_positions
    that exists in site_to_aa. For each AA in site_to_aa[pos], enumerate all codons for that AA.

    - If include_wt=False: skips the WT amino acid (assumed to be the first key in insertion order).
    - Deduplicates identical DNA sequences (keeps the first encountered label for that sequence).

    Returns:
      (seqs, labels)
        seqs[i] corresponds to labels[i] (values from site_to_aa[pos][aa])
    """
    dna = dna_seq.upper()
    L = len(dna)

    table = CodonTable.unambiguous_dna_by_id[genetic_code_id]

    # AA -> codons (include stop codons for '*')
    aa_to_codons: Dict[str, List[str]] = {}
    for codon, aa in table.forward_table.items():
        aa_to_codons.setdefault(aa, []).append(codon)
    for codon in table.stop_codons:
        aa_to_codons.setdefault("*", []).append(codon)

    # stable codon order
    for aa in aa_to_codons:
        aa_to_codons[aa] = sorted(aa_to_codons[aa])

    out_seqs: List[str] = []
    out_labels: List[Union[List[Any], Tuple[Any, ...]]] = []
    seen = set()

    for aa_pos in aa_positions:
        if aa_pos not in site_to_aa:
            continue

        nt0 = cds_start_0based + (int(aa_pos) - 1) * 3
        if nt0 < 0 or nt0 + 3 > L:
            continue

        old_codon = dna[nt0:nt0 + 3]
        if any(b not in "ACGT" for b in old_codon):
            continue

        aa_dict = site_to_aa[aa_pos]
        aa_items = list(aa_dict.items())
        if not aa_items:
            continue

        # WT is assumed to be the first key (insertion order)
        if include_wt:
            items_to_use = aa_items
        else:
            items_to_use = aa_items[1:]  # skip WT

        for aa, label_vec in items_to_use:
            aa = aa.upper()
            codons = aa_to_codons.get(aa, [])
            if not codons:
                continue

            for new_codon in codons:
                mut_dna = dna[:nt0] + new_codon + dna[nt0 + 3:]

                if mut_dna in seen:
                    continue
                seen.add(mut_dna)

                out_seqs.append(mut_dna)
                out_labels.append(label_vec)

    return out_seqs, out_labels

def parse_mut_positions(df: pd.DataFrame, mutation_col: str | None = None) -> dict[int, list[str]]:
    """
    Returns {pos: [AA_to1, AA_to2, ...]} from mutations like A102K, A102L.
    (only mutant amino acids, WT is ignored)
    """
    if mutation_col is None:
        mutation_col = df.columns[0]

    out: dict[int, list[str]] = {}
    for mut in df[mutation_col].dropna().astype(str):
        m = _MUT_RE.match(mut.strip())
        if m is None:
            raise ValueError(f"Bad mutation format: {mut}")
        pos = int(m.group(2))
        aa_to = m.group(3).upper()
        out.setdefault(pos, [])
        if aa_to not in out[pos]:
            out[pos].append(aa_to)
    return out


def make_most_frequent_codon_mutants(
    aa2codons_freq: dict[str, list[tuple[float, str]]],
    dna_seq: str,
    cds_start_0based: int,
    df: pd.DataFrame,
    aa_positions: list[int],
    fn_col: str,
    mutation_col: str | None = None,
) -> tuple[list[str], list[float]]:
    if mutation_col is None:
        mutation_col = df.columns[0]

    dna = dna_seq.upper()
    L = len(dna)
    pos_set = set(int(x) for x in aa_positions)

    mut_seqs: list[str] = []
    fn_scores: list[float] = []

    for _, row in df.iterrows():
        mut_raw = row.get(mutation_col)
        if pd.isna(mut_raw):
            continue

        # skip if fn score missing
        v = row.get(fn_col, None)
        if v is None or pd.isna(v):
            continue
        v = float(v)

        mut = str(mut_raw).strip()
        m = _MUT_RE.match(mut)
        if m is None:
            raise ValueError(f"Bad mutation format: {mut}")

        pos = int(m.group(2))
        if pos not in pos_set:
            continue

        aa_to = m.group(3).upper()
        if aa_to not in aa2codons_freq or not aa2codons_freq[aa_to]:
            raise ValueError(f"No codon frequencies for AA '{aa_to}' (mutation={mut})")

        _, best_codon = max(aa2codons_freq[aa_to], key=lambda x: x[0])
        best_codon = best_codon.upper()

        nt0 = cds_start_0based + (pos - 1) * 3
        if nt0 < 0 or nt0 + 3 > L:
            raise ValueError(
                f"AA pos {pos} out of range for dna length={L} with cds_start={cds_start_0based}"
            )

        new_dna = dna[:nt0] + best_codon + dna[nt0 + 3:]
        mut_seqs.append(new_dna)
        fn_scores.append(v)

    return mut_seqs, fn_scores

biodata = '/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT/phage/biodata'
fasta_path = f'{biodata}/sequence.fasta'
mutations_path = f'{biodata}/mutations_effect.tsv'


df_filter = {"effect": ["neg", "no_effect", "pos_or_neutral"], "single_mut": [1]}
#df_filter = {"effect": ["pos_or_neutral"], "single_mut": [1]}
mutation_df = load_and_filter_mutation_csv(mutations_path,
                                            metadata_keep=df_filter,
                                            first_library_col='lib1_DH10_pooled_F',
                                            sep='\t')


aa_positions = unique_mutation_sites(mutation_df)
print(f"aa_positions = {len(aa_positions)}")
dict_path = f"{biodata}/rbp_codon_usage.pkl"
with open(dict_path, 'rb') as f:
    aa2codons_freq = pickle.load(f)
 
fasta = pysam.Fastafile(fasta_path)

chrom = 'V01146.1'
start = 34624-1
end = 36285
cds_start = 0
sequence = fasta.fetch(reference=chrom, start=start, end=end).upper()


test_list = [i for i in range(1, len(sequence)//3 + 1, 10)]
train_list = [i for i in range(1, len(sequence)//3 + 1)]
train_list = list(set(train_list) - set(test_list))

#print(f"len_dna = {len(sequence)}")
#print(f"len_protein = {len(sequence)//3}")
#print(f"train_list = {len(train_list)}")
#print(f"test_list = {len(test_list)}")

test_mut_seqs, test_fn_scores = make_most_frequent_codon_mutants(
    aa2codons_freq=aa2codons_freq,
    dna_seq=sequence,
    cds_start_0based=cds_start,
    df=mutation_df,
    aa_positions=test_list,
    fn_col="lib1_DH10_pooled_F",
    mutation_col="Mutation",
)

train_mut_seqs, train_fn_scores = make_most_frequent_codon_mutants(
    aa2codons_freq=aa2codons_freq,
    dna_seq=sequence,
    cds_start_0based=cds_start,
    df=mutation_df,
    aa_positions=train_list,
    fn_col="lib1_DH10_pooled_F",
    mutation_col="Mutation",
)

y_min=1e-2
y_max=1e2
alpha=1.0

test_mut_seqs.append(sequence)
test_fn_scores.append(1.0)

def _transform_y(y: float, y_min=1e-2, y_max=1e2, alpha=1.0) -> torch.Tensor:
    y = min(max(y, y_min), y_max)
    logy = math.log10(y)

    z = alpha * logy
    return torch.sigmoid(torch.tensor(z, dtype=torch.float32))

train_fn_scores = [_transform_y(i) for i in train_fn_scores]
test_fn_scores = [_transform_y(i) for i in test_fn_scores]

print(len(train_fn_scores))
print(len(test_fn_scores))
#raise

def plot_distribution(data, save_path):

    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=20, range=(0, 1), edgecolor='black', alpha=0.7)
    #plt.title('')
    plt.xlabel('Fn-score transformed')
    plt.ylabel('Freq')
    #plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

#print(np.sort(test_fn_scores))
#raise
#plot_distribution(train_fn_scores, '/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT/finetuning/datasets/test_dataset/train.png')
#plot_distribution(test_fn_scores, '/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT/finetuning/datasets/test_dataset/test.png')
#raise

output_path = '/home/jovyan/shares/SR003.nfs2/caduseus_artem/phageml/ModernBERT/downstream_tasks/phageml/datasets/simple_test'
train_output = f'{output_path}/train_nonzero.hdf5'
with h5py.File(train_output, "a") as file:
    #index_sample = 0
    for i, fn in tqdm(enumerate(train_fn_scores), total=len(train_fn_scores), desc='Processing train'):
        group = file.create_group(f"sample_{i}") 
        group.attrs['seq'] = train_mut_seqs[i]

        group.create_dataset("fn_transform", data=np.array([fn], dtype=np.float64))
        #index_sample += 1

test_output = f'{output_path}/test_nonzero.hdf5'
with h5py.File(test_output, "a") as file:
    #index_sample = 0
    for i, fn in tqdm(enumerate(test_fn_scores), total=len(test_fn_scores), desc='Processing train'):
        group = file.create_group(f"sample_{i}") 
        group.attrs['seq'] = test_mut_seqs[i]

        group.create_dataset("fn_transform", data=np.array([fn], dtype=np.float64))
        #index_sample += 1

