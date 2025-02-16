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
from rocketshp.data.mdcath import convert_to_mdtraj
from rocketshp.features import esm3_vqvae
from rocketshp.esm3 import get_structure_vae
from rocketshp.trajectory import frame_to_chain

import os
import psutil
import gc

#%% Define file paths
logger.info("Defining file paths")

MDCATH_DATA_DIR = config.RAW_DATA_DIR / "mdcath"
MDCATH_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "mdcath"
MDCATH_FOLDSEEK_CLUSTERS_FILE = config.PROCESSED_DATA_DIR / "mdcath/foldseek_atlas_0.2_cluster.tsv"

mdcath_files = list(MDCATH_DATA_DIR.glob("mdcath_dataset_*.h5"))
N_REPS = 5
TEMPS = [320, 348, 379, 413, 450]
RANDOM_STATE = 42
OVERWRITE_H5 = False
# I_START = int(sys.argv[1])
# I_STOP = int(sys.argv[2])
# I_START = 0
# I_STOP = 20

#%% Define process function

def compute_trajectory_derivatives(pdb_id):
    # process = psutil.Process(os.getpid())
    # logger.info(f"Worker {os.getpid()} initial memory: {process.memory_info().rss / 1024 / 1024} MB")
    # logger.info(f"Starting batch of {len(pdb_id)}")
    mdcath_f, rep, temp = pdb_id
    pdb_code = mdcath_f.split("_")[-1].split(".")[0]
    logger.info(f"Processing {pdb_code}:{rep}:{temp}")
    # logger.info(f"Worker {os.getpid()} memory before trajectory: {process.memory_info().rss / 1024 / 1024} MB")
    
    # Load trajectory
    traj = convert_to_mdtraj(mdcath_f, temp, rep)
    
    # logger.info(f"Worker {os.getpid()} memory after load: {process.memory_info().rss / 1024 / 1024} MB")
    
    traj = normalize(traj, ca_only=True)
    rmsf = compute_rmsf(traj, normalized=True, ca_only=True)
    # logger.info(f"Worker {os.getpid()} memory after RMSF: {process.memory_info().rss / 1024 / 1024} MB")
    
    contacts = compute_contacts(traj, scheme="ca", ignore_nonprotein=True, normalized=True, ca_only=True)
    # logger.info(f"Worker {os.getpid()} memory after contacts: {process.memory_info().rss / 1024 / 1024} MB")
    
    ca_dist = contacts[0]
    autocorr = compute_autocorrelation(traj, precomputed_contacts=contacts, normalized=True, ca_only=True)
    # logger.info(f"Worker {os.getpid()} memory after autocorr: {process.memory_info().rss / 1024 / 1024} MB")

    r = {
        "pdb_code": pdb_code,
        "rep": rep,
        "temp": temp,
        "h5_file": mdcath_f,
        "rmsf": rmsf.copy(),
        "ca_dist": ca_dist.copy(),
        "autocorr": autocorr.copy(),
    }
    
    # Explicit cleanup
    del traj, contacts, rmsf, ca_dist, autocorr
    gc.collect()  # Force garbage collection
    # logger.info(f"Worker {os.getpid()} memory after cleanup: {process.memory_info().rss / 1024 / 1024} MB")
    # logger.info(f"Worker {os.getpid()} has {len(gc.get_objects())} total objects")
    
    return r

def compute_batched_trajectory_derivatives(pdb_id):
    # process = psutil.Process(os.getpid())
    # logger.info(f"Worker {os.getpid()} initial memory: {process.memory_info().rss / 1024 / 1024} MB")
    # logger.info(f"Starting batch of {len(pdb_id)}")
    
    results = []
    for pid in pdb_id:
        r = compute_trajectory_derivatives(pid)
        results.append(r)
    return results


#%% Define all reps
pdb_reps = [(str(mdc_f), rep, temp) for mdc_f in mdcath_files for rep in range(N_REPS) for temp in TEMPS][I_START:I_STOP]
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
    results.append(compute_trajectory_derivatives(pdb_id))
#%% # Create HuggingFace dataset

logger.info("Creating HuggingFace dataset")
ds = Dataset.from_dict(invert_dict(results))
logger.info(f"Dataset created with {len(ds)} samples")
out_path = MDCATH_PROCESSED_DATA_DIR / f"mdcath_derivatives_v2_{I_START}_{I_STOP}"
logger.info(f"Saving dataset to {out_path}")
ds.save_to_disk(str(out_path))
# logger.info(f"Saved to {out_path}")

#%% Join data sets from different initializations
logger.info("Joining datasets")
from datasets import concatenate_datasets
subsets = [
    (0, 16200),
    (16200, 32400),
    (32400, 48600),
    (48600, 64800),
    (64800, 81000),
    (81000, 97200),
    (97200, 113400),
    (113400, 129600),
]
all_ds = []
for (s0, s1) in subsets:
    ds = Dataset.load_from_disk(str(MDCATH_PROCESSED_DATA_DIR / f"mdcath_derivatives_v2_{s0}_{s1}"))
    all_ds.append(ds)

all_ds = concatenate_datasets(all_ds)
all_ds.save_to_disk(str(MDCATH_PROCESSED_DATA_DIR / "mdcath_derivatives_v2"))
# %%
