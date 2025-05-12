#%%
import re
import time
import mdtraj as md
from tqdm import tqdm
from pathlib import Path
from bioemu.sample import main as sample
from loguru import logger

num_samples = 100
data_root = Path("/mnt/home/ssledzieski/Projects/rocketshp/data")
output_root = Path("/mnt/home/ssledzieski/GitHub/bioemu/rshp_atlas_results_100")
atlas_data = data_root / "raw/atlas"


#%% Get all atlas pdb files
logger.info("Getting all atlas pdbs...")
pdb_files = [f for f in atlas_data.glob("**/*.pdb") if f.is_file() and re.search(r"...._.\.pdb", f.name)]
logger.info(f"Found {len(pdb_files)} pdbs.")

# %% Get all sequences from the pdb files
from biotite.sequence import ProteinSequence

sequences = []

if (output_root / "sequences.fasta").exists():

    # Read sequences from the file
    with open(output_root / "sequences.fasta", "r") as f:
        for line in f:
            if line.startswith(">"):
                pdb_key = line[1:].strip()
                seq = next(f).strip()
                sequences.append(seq)
                logger.info(f"Loaded sequence for {pdb_key} from file.")

else:
    logger.info("Getting all sequences from the pdb files...")
    for pdb_file in tqdm(pdb_files):
        t = md.load(str(pdb_file))

        # Get sequence from trajectory topology
        residue_names = [residue.name for residue in t.topology.residues]

        # Convert 3-letter codes to 1-letter amino acid codes
        aa_sequence = ''.join([ProteinSequence.convert_letter_3to1(res) for res in residue_names])
        sequences.append(aa_sequence)

    # Write sequences to a file
    with open(output_root / "sequences.fasta", "w+") as f:
        for pdb_f, seq in zip(pdb_files, sequences):
            f.write(f">{pdb_f.stem}\n{seq}\n")

# %% Run bioemu for each sequence
cache_dir = Path(f"/tmp/rocketshp/bioemu_atlas_{num_samples}_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

logger.info("Running bioemu for each sequence...")
for pdb_fi, seq in tqdm(zip(pdb_files, sequences)):
    time.sleep(0.1) # to avoid spamming the MSA server-- not counted in running time
    
    pdb_key = pdb_fi.stem
    if Path(f"{output_root}/{pdb_key}_{num_samples}/time_log.txt",).exists():
        logger.info(f"Already processed {pdb_fi.stem} with {num_samples} samples.")
        continue
    
    start_time = time.time()
    try:
        sample(
            sequence=seq,
            num_samples=num_samples,
            output_dir=f"{output_root}/{pdb_key}_{num_samples}",
            cache_embeds_dir=str(cache_dir),
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        with open(f"{output_root}/{pdb_key}_{num_samples}/time_log.txt", "w") as f:
            f.write(f"Model inference time: {elapsed_time:.5f}\n")

    except Exception as e:
        logger.error(f"Error processing {pdb_fi.stem}: {e}")
        with open(f"{output_root}/{pdb_key}_{num_samples}/error.txt", "w") as f:
            f.write(f"Error: {e}\n")
        continue
    
# %%
