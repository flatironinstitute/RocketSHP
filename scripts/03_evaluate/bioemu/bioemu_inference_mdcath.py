#%%
import re
import time
import mdtraj as md
import h5py
from tqdm import tqdm
from pathlib import Path
from bioemu.sample import main as sample

num_samples = 100
data_root = Path("/mnt/home/ssledzieski/Projects/rocketshp/data")
output_root = Path("/mnt/home/ssledzieski/GitHub/bioemu/rshp_results_100")
mdcath_root  = data_root / "processed/mdcath"

#%% Get all mdcath 
mdcath_dirs = [f for f in mdcath_root.glob("*") if f.is_dir()]
mdcath_pdbs = [x / x.with_suffix(".pdb").name for x in mdcath_dirs]

# %% Get all sequences from the pdb files
from biotite.sequence import ProteinSequence

sequences = []
for pdb_file in tqdm(mdcath_pdbs):
    t = md.load(str(pdb_file))

    # Get sequence from trajectory topology
    residue_names = [residue.name for residue in t.topology.residues]

    # Convert 3-letter codes to 1-letter amino acid codes
    aa_sequence = ''.join([ProteinSequence.convert_letter_3to1(res) for res in residue_names])
    sequences.append(aa_sequence)

# %% Run bioemu for each sequence
with open(output_root / f"bioemu_{num_samples}_time_results.txt", "w") as f:
    for pdb_fi, seq in tqdm(zip(mdcath_pdbs, sequences)):
        time.sleep(2) # to avoid spamming the MSA server-- not counted in running time
        start_time = time.time()
        pdb_key = pdb_fi.stem
        sample(
            sequence=seq,
            num_samples=num_samples,
            output_dir=f"{output_root}/{pdb_key}_{num_samples}",
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        f.write(f"{pdb_key}\t{len(seq)}\t{elapsed_time:.2f}\n")
        f.flush()
# %%
