# %% Imports
import pandas as pd
import numpy as np
from loguru import logger
from biotite.sequence.io.fasta import FastaFile

# %% Load data
rxdb_file = "/mnt/home/ssledzieski/GitHub/Dyna-1/data/RelaxDB_datasets/RelaxDB_with_other_metrics_22jan2025.json.zip"
output_file = "/mnt/home/ssledzieski/GitHub/Dyna-1/data/RelaxDB_datasets/RelaxDB_22jan2025.fasta"

d = pd.read_json(rxdb_file)
for col in d.columns:
    if isinstance(d[col].iloc[0],list):
        d[col] = [np.asarray(x,dtype=float) for x in d[col]]

logger.info(len(d))

# %% Melt to per residue
m=[]

for _, row in d.iterrows():
    for i in range(len(row['sequence'])):
        start = row['missing_peaks'].find('A')
        end = row['missing_peaks'].rfind('A')
        if i >= start and i <= end: # and i >= int(0.05*row['seq length']) and i <= int(0.95*row['seq length']):
            if row['sequence'][i]=='P':
                label='Pro'
            elif row['missing_peaks'][i]=='.':
                label='missing'
            elif row['label'][i] == 'x':
                label='na'
            elif row['label'][i] == 'v':
                label='low'
            elif row['label'][i]=='^':
                label='high'
            elif row['label'][i] == 'b':
                label='high'
            else:
                label='mid'
        else:
            label='term'

        dssp_dct={'C': 'Loop','E':'Sheet','H':'Helix', ' ': 'Loop'}

        m.append({'AA': row['sequence'][i],
                  'seqpos': i,
                  'entry_ID': row['entry_ID'],
                  'Classification': row['Classification'],
                  'DSSP': dssp_dct[row['DSSP'][i]],
                  'R2_R1': row['R2_R1'][i],
                  'dR2': row['dR2'][i],
                  'cons': row['cons_BLOSUM62'][i],
                  'coverage': row['coverage'][i],
                    'pLDDT': row['pLDDT'][i],
                  'SASA': row['SASA'][i],
                  'NOE': row['NOE'][i],
                  'label':label})

melt = pd.DataFrame.from_records(m)

# %% Convert to fasta file
ff = FastaFile()
for _, r in d.iterrows():
    idx = r["entry_ID"]
    seq = r['sequence']
    ff[str(idx)] = seq

ff.write(output_file)
logger.info(f"Fasta file written to {output_file}")

# %% Convert to individual fasta files
for header, seq in ff.items():
    output_file = f"/mnt/home/ssledzieski/GitHub/Dyna-1/data/RelaxDB_datasets/fasta_files/{header}.fasta"
    with open(output_file, "w") as f:
        f.write(f">{header}\n")
        f.write(f"{seq}\n")
    logger.info(f"Fasta file written to {output_file}")

# %%
