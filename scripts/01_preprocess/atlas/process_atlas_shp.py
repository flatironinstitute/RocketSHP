#%% # Load packages
import mdtraj as md
import torch
import sys
from datasets import Dataset
from loguru import logger
from tqdm import tqdm

from rocketshp import config
from rocketshp.parallel import parallel_pool
from rocketshp.trajectory import (
    compute_autocorrelation,
    compute_contacts,
    compute_rmsf,
    normalize,
)
from rocketshp.features import esm3_vqvae
from rocketshp.esm3 import get_structure_vae
from rocketshp.trajectory import frame_to_chain

import os
import psutil
import gc

#%% Define file paths
logger.info("Defining file paths")

ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
ATLAS_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "atlas"
ATLAS_FOLDSEEK_CLUSTERS_FILE = config.PROCESSED_DATA_DIR / "atlas/foldseek_atlas_0.2_cluster.tsv"

xtc_files = list(ATLAS_DATA_DIR.glob("*/*.xtc"))
pdb_files = list(ATLAS_DATA_DIR.glob("*/*.pdb"))
pdb_files = [i for i in pdb_files if ".ca.pdb" not in i.name]
N_REPS = 3
RANDOM_STATE = 42
OVERWRITE_H5 = False
I_START = int(sys.argv[1])
I_STOP = int(sys.argv[2])

#%% Define process function

def compute_trajectory_shp(pdb_id, start=0, end=None, stride=100):
    # process = psutil.Process(os.getpid())
    # logger.info(f"Worker {os.getpid()} initial memory: {process.memory_info().rss / 1024 / 1024} MB")
    # logger.info(f"Starting batch of {len(pdb_id)}")
    pdb_code, rep = pdb_id
    logger.info(f"Processing {pdb_code}:{rep}")
    # logger.info(f"Worker {os.getpid()} memory before trajectory: {process.memory_info().rss / 1024 / 1024} MB")
    
    # Load trajectory
    xtc_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}_prod_R{rep}_fit.xtc"
    pdb_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}.pdb"
    traj = md.load(str(xtc_f), top=pdb_f)
    # logger.info(f"Worker {os.getpid()} memory after load: {process.memory_info().rss / 1024 / 1024} MB")
    
    traj = normalize(traj, ca_only=False)
    traj = traj[start:end:stride]
    
    struct_encoder = get_structure_vae()
    shp = []
    with torch.inference_mode():
        for frame in tqdm(traj):
            chain = frame_to_chain(frame)
            frame_q = esm3_vqvae(chain, struct_encoder, stage="quantized")
            # logger.info(frame_q)
            shp.append(frame_q)
    shp = torch.stack(shp)

    # Explicit cleanup
    del traj
    gc.collect()  # Force garbage collection
    # logger.info(f"Worker {os.getpid()} memory after cleanup: {process.memory_info().rss / 1024 / 1024} MB")
    # logger.info(f"Worker {os.getpid()} has {len(gc.get_objects())} total objects")
    
    return {
        "pdb_code": pdb_code,
        "rep": rep,
        "xtc_file": xtc_f.stem,
        "pdb_file": pdb_f.stem,
        "shp": shp,
    }


#%% Define all reps
pdb_codes = [pdb_f.stem for pdb_f in pdb_files]
pdb_reps = [(pdb_code, rep) for pdb_code in pdb_codes for rep in range(1, N_REPS + 1)][I_START:I_STOP]
TOTAL_JOBS = len(pdb_reps)

#%% Compute values in parallel
# N_JOBS = 50
# BATCH_SIZE = 1

# logger.info(f"Computing {TOTAL_JOBS} reps in parallel ({N_JOBS} workers, batch size {BATCH_SIZE})")
# results = parallel_pool(pdb_reps, compute_batched_trajectory_derivatives, n_jobs=N_JOBS, batch_size=BATCH_SIZE, report_every=1)
# logger.info(f"Finished processing {len(results)} reps")

#%% Invert Dicts
def invert_dict(l):
    """
    Convert from list of dicts to dict of lists, where the key is the joined "pdb_code" and "rep" keys of the inner dictionary
    """
    out_dict = {}
    keys = l[0].keys()
    for key in keys:
        out_dict[key] = [i[key] for i in l]
    return out_dict


#%% Just do serial to not deal with memory issues
results = []
logger.info(f"Computing {TOTAL_JOBS} reps")
for pdb_id in tqdm(pdb_reps):
    # results.append(compute_trajectory_derivatives(pdb_id))
    results.append(compute_trajectory_shp(pdb_id))
#%% # Create HuggingFace dataset

logger.info("Creating HuggingFace dataset")
ds = Dataset.from_dict(invert_dict(results))
logger.info(f"Dataset created with {len(ds)} samples")
out_path = ATLAS_PROCESSED_DATA_DIR / f"atlas_shp_v2_{I_START}_{I_STOP}"
logger.info(f"Saving dataset to {out_path}")
ds.save_to_disk(str(out_path))
logger.info(f"Saved to {out_path}")

#%% Process data
# with h5py.File(ATLAS_PROCESSED_DATA_DIR / "atlas_processed.h5", "a") as h5file:



    # for pdb_f in tqdm(pdb_files, total=len(pdb_files)):
    #     pdb_code = pdb_f.stem

    #     h5file.require_group(pdb_code)

    #     for rep in range(1, N_REPS + 1):
    #         xtc_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}_prod_R{rep}_fit.xtc"
    #         if not xtc_f.exists():
    #             logger.warning(f"Missing {xtc_f}")
    #             continue

    #         # if f"{pdb_code}/R{rep}" in h5file:
    #             # logger.warning(f"Skipping {pdb_code}")
    #             # continue

    #         # traj = md.load_pdb(pdb_f)
    #         traj = md.load(str(xtc_f), top=pdb_f)
    #         traj.superpose(traj, 0)
    #         traj.center_coordinates()

            # update_h5_dataset(h5file, f"{pdb_code}/R{rep}/xyz", torch.from_numpy(traj.xyz))
            # update_h5_dataset(h5file, f"{pdb_code}/R{rep}/time", torch.from_numpy(traj.time))
            # update_h5_dataset(h5file, f"{pdb_code}/R{rep}/rmsf", md.rmsf(
                # traj, traj, 0, atom_indices=traj.top.select("name CA")
            # ))
            # update_h5_dataset(h5file, f"{pdb_code}/R{rep}/rmsd", md.rmsd(
                # traj, traj, 0, atom_indices=traj.top.select("name CA")
            # ))
            # update_h5_dataset(h5file, f"{pdb_code}/R{rep}/rg", md.compute_rg(traj))
            # update_h5_dataset(h5file, f"{pdb_code}/R{rep}/gyration", md.compute_gyration_tensor(traj))
            # update_h5_dataset(h5file, f"{pdb_code}/R{rep}/principal_moments", md.principal_moments(traj))
            # update_h5_dataset(h5file, f"{pdb_code}/R{rep}/ca_distances", md.geometry.squareform(
                # *md.compute_contacts(traj[0], scheme="ca", ignore_nonprotein=True)
            # ))

            # logger.info(f"Processed {pdb_code}:{rep}")
