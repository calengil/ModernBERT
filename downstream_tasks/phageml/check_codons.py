#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pysam
from Bio.Data import CodonTable
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pickle

def split_info_fields(info_record, list_records):
    """
    Add new columns from attribures
    """
    
    records = info_record.split(";")
    result = {}
    for record in records:
        record_data = record.split("=")
        result[record_data[0]] = record_data[1]
    
    list_records.append(result)

def prepare_gff(gff, chrom=['V01146.1']):
    col_names = [
        "seqid",
        "source",
        "type",
        "start",
        "end",
        "score",
        "strand",
        "phase",
        "attributes",
    ]
    
    data_gff = pd.read_csv(gff, sep="\t", names=col_names, header=None, comment="#")
    data_gff["start"] = (data_gff["start"] - 1)  # coodinates in GFF file are 1-based, convert to 0-based
    # data_gff["end"] = data_gff["end"] - 1 # do not substract from the end; intervals in GFF are closed, but now we can consider
    #                               them as half-opened intervals
    
    data_gff['lens'] = data_gff['end'] - data_gff['start']
        
       
    data = data_gff[(data_gff["type"].isin(['CDS'])) & (data_gff["seqid"].isin(chrom))]
    data.reset_index(drop=True, inplace=True)    
    new_col_list = []
    data["attributes"].apply(split_info_fields, list_records=new_col_list)
    new_cols = pd.DataFrame.from_dict(new_col_list)
    data = pd.concat((data, new_cols), axis=1)
    
    id_indexed = data.set_index("ID")
    
    #gene_list = data['ID'].to_list()

    return data, id_indexed

def search_parts(transcript_content, part):
    
    part_cotent = transcript_content[transcript_content["type"] == part]
        
    starts_part = np.array(part_cotent['start'].tolist())
    ends_part = np.array(part_cotent['end'].tolist())

    return list(zip(starts_part, ends_part))

def codon_usage(cds_list):
    table = CodonTable.unambiguous_dna_by_name["Standard"]

    aa_to_all_codons = defaultdict(list)
    for codon, aa in table.forward_table.items():
        aa_to_all_codons[aa].append(codon)

    aa_codon_counts = defaultdict(Counter)

    for cds in cds_list:
        #print(cds)
        #print(cds[:3], cds[-3:])
        #print(len(cds))
        if len(cds) % 3 != 0:
            #print(cds_list.index(cds))
            raise
            #continue

        for i in range(0, len(cds), 3):
            codon = cds[i:i+3].upper()
            if len(codon) != 3:
                break
            
            aa = table.forward_table.get(codon)
            if aa:
                aa_codon_counts[aa][codon] += 1

    aa_codon_freq_dict = {}
    for aa in sorted(aa_codon_counts.keys()):
        total = sum(aa_codon_counts[aa].values())
        all_codons = sorted(aa_to_all_codons[aa])
        freq_list = []
        for codon in all_codons:
            count = aa_codon_counts[aa].get(codon, 0)
            frequency = count / total if total > 0 else 0
            freq_list.append((frequency, codon))
        freq_list.sort(key=lambda x: x[0], reverse=True)
        aa_codon_freq_dict[aa] = freq_list

    return aa_codon_freq_dict

def draw_codon_heatmap(aa_codon_freq_dict, output_path):
    """
    Draws a codon usage heatmap in the standard genetic code table layout with grouped amino acids.
    - aa_codon_freq_dict: dict where key is amino acid, value is list of (frequency, codon) tuples.
    - output_path: path to save the figure (e.g., 'heatmap.png').
    Frequencies are from 0 to 1 (proportion per amino acid).
    """
    # Get standard table
    table = CodonTable.unambiguous_dna_by_name["Standard"]
    
    # Bases order (DNA: T instead of U)
    bases = 'TCAG'
    
    # Create frequency matrix: 4 rows (second base), 16 columns (first base groups x third base)
    freq_matrix = np.zeros((4, 16))
    codon_matrix = [['' for _ in range(16)] for _ in range(4)]
    aa_matrix = [['' for _ in range(16)] for _ in range(4)]
    
    # Mapping from codon to frequency and aa
    codon_to_freq = {}
    codon_to_aa = {}
    for aa, lst in aa_codon_freq_dict.items():
        for freq, codon in lst:
            codon_to_freq[codon] = freq
            codon_to_aa[codon] = aa
    
    # Codon to position mapping and fill matrix
    codon_to_pos = {}
    for i, second in enumerate(bases):
        for first_idx in range(4):
            first = bases[first_idx]
            for third_idx in range(4):
                third = bases[third_idx]
                codon = first + second + third
                j = first_idx * 4 + third_idx
                codon_to_pos[codon] = (i, j)
                codon_matrix[i][j] = codon
                aa_matrix[i][j] = codon_to_aa.get(codon, '*')  # Use '*' for stops
                # Set frequency (0 if not present)
                freq_matrix[i, j] = codon_to_freq.get(codon, 0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(24, 8))  # Larger for better readability
    im = ax.imshow(freq_matrix, cmap='YlGnBu', vmin=0, vmax=1)
    
    # Colorbar
    fig.colorbar(im, ax=ax, label='Frequency (proportion)', shrink=0.5)
    
    # Y-axis: second base
    ax.set_yticks(range(4))
    ax.set_yticklabels(list(bases), fontsize=14)
    ax.set_ylabel('Second base', fontsize=16)
    
    # X-axis: first base (major ticks at centers)
    major_xticks = [1.5, 5.5, 9.5, 13.5]  # Centers of groups
    ax.set_xticks(major_xticks, minor=False)
    ax.set_xticklabels(list(bases), minor=False, fontsize=14)
    ax.set_xlabel('First base', fontsize=16)
    
    # Minor xticks for third base
    minor_xticks = np.arange(0.5, 16, 1)  # Centers of each column
    ax.set_xticks(minor_xticks, minor=True)
    minor_labels = list(bases) * 4
    ax.set_xticklabels(minor_labels, minor=True, fontsize=10)
    ax.xaxis.set_ticks_position('top')  # Move third base labels to top for standard look
    ax.xaxis.set_label_position('top')
    
    # Grid lines: vertical between first base groups
    for x in range(4, 16, 4):
        ax.axvline(x - 0.5, color='black', lw=2)
    # Horizontal lines between rows
    for y in range(1, 4):
        ax.axhline(y - 0.5, color='black', lw=2)
    
    # Add aa, codon, and frequency text to each cell
    for i in range(4):
        for j in range(16):
            freq = freq_matrix[i, j]
            codon = codon_matrix[i][j]
            aa = aa_matrix[i][j]
            color = 'white' if freq > 0.5 else 'black'  # Adjust text color for visibility
            ax.text(j, i - 0.25, aa, ha="center", va="center", color=color, fontsize=10, fontweight='bold')
            ax.text(j, i, codon, ha="center", va="center", color=color, fontsize=10)
            ax.text(j, i + 0.25, f"{freq:.2f}", ha="center", va="center", color=color, fontsize=10)
    
    # Title
    ax.set_title('Codon Usage Heatmap (Standard Genetic Code Layout)', fontsize=18)
    
    # Save and close
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


biodata = '/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT/phage/biodata'
fasta = f'{biodata}/sequence.fasta'
gff = f'{biodata}/sequence.gff3'
chrom = 'V01146.1'
output_path = '/home/jovyan/shares/SR003.nfs2/caduseus_artem/phage/ModernBERT/output/codon_usage_T7_rbp.png'

fasta = pysam.Fastafile(fasta)

data, id_indexed = prepare_gff(gff)
cds_list = search_parts(data, 'CDS')

#sequences = [fasta.fetch(reference=chrom, start=start, end=end).upper() for (start, end) in cds_list]

start = 34624-1
end = 36285
sequences = [fasta.fetch(reference=chrom, start=start, end=end).upper()]


for seq in sequences:
    #print(len(seq) % 3)
    assert len(seq) % 3 == 0
#raise
codon_usage_dict = codon_usage(sequences)

save_path = f'{biodata}/rbp_codon_usage.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(codon_usage_dict, f)

#draw_codon_heatmap(codon_usage_dict, output_path)