#%% # Load packages
import mdtraj as md
from datasets import Dataset
from loguru import logger

from rocketshp import config
from rocketshp.parallel import parallel_pool
from rocketshp.trajectory import (
    compute_autocorrelation,
    compute_contacts,
    compute_rmsf,
    normalize,
)

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
pdb_files = [i for i in pdb_files if "ca" not in i.stem]
N_REPS = 3
RANDOM_STATE = 42
OVERWRITE_H5 = False

#%% Define process function

def compute_batched_trajectory_derivatives(pdb_id):
    # process = psutil.Process(os.getpid())
    # logger.info(f"Worker {os.getpid()} initial memory: {process.memory_info().rss / 1024 / 1024} MB")
    # logger.info(f"Starting batch of {len(pdb_id)}")
    
    results = []
    for pid in pdb_id:
        pdb_code, rep = pid
        logger.info(f"Processing {pdb_code}:{rep}")
        # logger.info(f"Worker {os.getpid()} memory before trajectory: {process.memory_info().rss / 1024 / 1024} MB")
        
        # Load trajectory
        xtc_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}_prod_R{rep}_fit.xtc"
        pdb_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}.pdb"
        traj = md.load(str(xtc_f), top=pdb_f)
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
            "xtc_file": xtc_f.stem,
            "pdb_file": pdb_f.stem,
            "rmsf": rmsf.copy(),
            "ca_dist": ca_dist.copy(),
            "autocorr": autocorr.copy(),
        }
        
        # Explicit cleanup
        del traj, contacts, rmsf, ca_dist, autocorr
        gc.collect()  # Force garbage collection
        # logger.info(f"Worker {os.getpid()} memory after cleanup: {process.memory_info().rss / 1024 / 1024} MB")
        # logger.info(f"Worker {os.getpid()} has {len(gc.get_objects())} total objects")
        
        results.append(r)
    return results


#%% Define all reps
pdb_codes = [pdb_f.stem for pdb_f in pdb_files]
pdb_reps = [(pdb_code, rep) for pdb_code in pdb_codes for rep in range(1, N_REPS + 1)]


#%% Compute values in parallel
N_JOBS = 30
BATCH_SIZE = 1
TOTAL_JOBS = len(pdb_reps)

logger.info(f"Computing {TOTAL_JOBS} reps in parallel ({N_JOBS} workers, batch size {BATCH_SIZE})")
results = parallel_pool(pdb_reps, compute_batched_trajectory_derivatives, n_jobs=60, batch_size=1, report_every=1)

#%% # Create HuggingFace dataset

logger.info("Creating HuggingFace dataset")
def invert_dict(l):
    """
    Convert from list of dicts to dict of lists, where the key is the joined "pdb_code" and "rep" keys of the inner dictionary
    """
    out_dict = {}
    keys = l[0].keys()
    for key in keys:
        out_dict[key] = [i[key] for i in l]
    return out_dict

ds = Dataset.from_dict(invert_dict(results))
logger.info(f"Dataset created with {len(ds)} samples")
logger.info(f"Saving dataset to {ATLAS_PROCESSED_DATA_DIR / 'atlas_derivatives_v2'}")
ds.save_to_disk(ATLAS_PROCESSED_DATA_DIR / "atlas_derivatives_v2")

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
