#%%
import re
import time
import mdtraj as md
from tqdm import tqdm
from pathlib import Path
from bioemu.sample import main as sample

num_samples = 100
data_root = Path("/mnt/home/ssledzieski/Projects/rocketshp/data")
output_root = Path("/mnt/home/ssledzieski/GitHub/bioemu/rshp_results_100")
atlas_data = data_root / "raw/atlas"


#%% Get all atlas pdb files
pdb_files = [f for f in atlas_data.glob("**/*.pdb") if f.is_file() and re.search(r"...._.\.pdb", f.name)]

# %% Get all sequences from the pdb files
from biotite.sequence import ProteinSequence

sequences = []
for pdb_file in tqdm(pdb_files):
    t = md.load(str(pdb_file))

    # Get sequence from trajectory topology
    residue_names = [residue.name for residue in t.topology.residues]

    # Convert 3-letter codes to 1-letter amino acid codes
    aa_sequence = ''.join([ProteinSequence.convert_letter_3to1(res) for res in residue_names])
    sequences.append(aa_sequence)

# %% Run bioemu for each sequence
with open(output_root / f"bioemu_{num_samples}_time_results.txt", "w") as f:
    for pdb_fi, seq in tqdm(zip(pdb_files, sequences)):
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
