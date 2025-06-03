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

# %% prioritize test set data first
with open("/mnt/home/ssledzieski/Projects/rocketshp/data/processed/mdcath_test_set.txt", "r") as f:
    mdcath_test_list = [l.strip() for l in f.readlines()]

mdcath_pdbs_test_first = sorted(mdcath_pdbs, key=lambda x: x.stem in mdcath_test_list, reverse=True)

for i in range(622):
    assert mdcath_pdbs_test_first[i].stem in mdcath_test_list, f"Test set not prioritized correctly: {mdcath_pdbs_test_first[i].stem} not in test set"

# %% Get all sequences from the pdb files
sequences = {}

if (output_root / "sequences.fasta").exists():
    with open(output_root / "sequences.fasta", "r") as f:
        for line in f:
            if line.startswith(">"):
                pdb_key = line[1:].strip()
                seq = next(f).strip()
                sequences[pdb_key] = seq
    logger.info(f"Loaded sequences from {output_root / 'sequences.fasta'}.")

else:
    logger.info("Getting all sequences from the pdb files...")
    for pdb_file in tqdm(mdcath_pdbs_test_first):
        t = md.load(str(pdb_file))
        pdb_key = pdb_file.stem

        # Get sequence from trajectory topology
        residue_names = [residue.name for residue in t.topology.residues]

        # Convert 3-letter codes to 1-letter amino acid codes
        aa_sequence = "".join(
            [ProteinSequence.convert_letter_3to1(res) for res in residue_names]
        )
        sequences[pdb_key] = aa_sequence

        # Write sequences to a file
        with open(output_root / "sequences.fasta", "w+") as f:
            for pdb_k, seq in sequences.items():
                f.write(f">{pdb_k}\n{seq}\n")

# %% Run bioemu for each sequence
cache_dir = Path(f"/tmp/rocketshp/bioemu_mdcath_{num_samples}_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

logger.info("Running bioemu for each sequence...")
sequence_list = [sequences[k.stem] for k in mdcath_pdbs_test_first]
for i, (pdb_fi, seq) in enumerate(tqdm(zip(mdcath_pdbs_test_first, sequence_list),total=len(mdcath_pdbs_test_first))):
    pdb_key = pdb_fi.stem
    if Path(
        f"{output_root}/{pdb_key}_{num_samples}/time_log.txt",
    ).exists():
        if pdb_key in mdcath_test_list:
            IN_TEST = "(in test)"
        else:
            IN_TEST = ""
        logger.info(f"Already processed {pdb_fi.stem} {IN_TEST} with {num_samples} samples (sample {i}).")
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
