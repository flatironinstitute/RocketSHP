# %%
import time
from pathlib import Path

import mdtraj as md
from bioemu.sample import main as sample
from biotite.sequence import ProteinSequence
from loguru import logger
from tqdm import tqdm

num_samples = 100
data_root = Path("/mnt/home/ssledzieski/Projects/rocketshp/data")
output_root = Path("/mnt/home/ssledzieski/GitHub/bioemu/rshp_mdcath_results_100")
output_root.mkdir(parents=True, exist_ok=True)
mdcath_root = data_root / "processed/mdcath"

# %% Get all mdcath
logger.info("Getting all mdcath pdbs...")
mdcath_dirs = [f for f in mdcath_root.glob("*") if f.is_dir() and len(f.stem) == 7]
mdcath_pdbs = [x / x.with_suffix(".pdb").name for x in mdcath_dirs]
logger.info(f"Found {len(mdcath_pdbs)} mdcath pdbs.")

# %% Get all sequences from the pdb files
sequences = []
logger.info("Getting all sequences from the pdb files...")
for pdb_file in tqdm(mdcath_pdbs):
    t = md.load(str(pdb_file))

    # Get sequence from trajectory topology
    residue_names = [residue.name for residue in t.topology.residues]

    # Convert 3-letter codes to 1-letter amino acid codes
    aa_sequence = "".join(
        [ProteinSequence.convert_letter_3to1(res) for res in residue_names]
    )
    sequences.append(aa_sequence)

# %% Run bioemu for each sequence
cache_dir = Path(f"/tmp/rocketshp/bioemu_mdcath_{num_samples}_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

logger.info("Running bioemu for each sequence...")
for pdb_fi, seq in tqdm(zip(mdcath_pdbs, sequences)):
    pdb_key = pdb_fi.stem
    if Path(
        f"{output_root}/{pdb_key}_{num_samples}/time_log.txt",
    ).exists():
        logger.info(f"Already processed {pdb_fi.stem} with {num_samples} samples.")
        continue

    # time.sleep(2)  # to avoid spamming the MSA server-- not counted in running time
    start_time = time.time()
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

        # f.write(f"{pdb_key}\t{len(seq)}\t{elapsed_time:.2f}\n")
        # f.flush()
# %%
