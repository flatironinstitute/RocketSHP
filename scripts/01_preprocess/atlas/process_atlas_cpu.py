# %% # Load packages
import gc
import sys

import mdtraj as md
from datasets import Dataset
from loguru import logger
from tqdm import tqdm

from rocketshp import config
from rocketshp.trajectory import (
    compute_autocorrelation,
    compute_contacts,
    compute_rmsf,
    normalize,
)

# %% Define file paths
logger.info("Defining file paths")

ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
ATLAS_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "atlas"
ATLAS_FOLDSEEK_CLUSTERS_FILE = (
    config.PROCESSED_DATA_DIR / "atlas/foldseek_atlas_0.2_cluster.tsv"
)

xtc_files = list(ATLAS_DATA_DIR.glob("*/*.xtc"))
pdb_files = list(ATLAS_DATA_DIR.glob("*/*.pdb"))
pdb_files = [i for i in pdb_files if ".ca.pdb" not in i.name]
N_REPS = 3
RANDOM_STATE = 42
OVERWRITE_H5 = False
I_START = int(sys.argv[1])
I_STOP = int(sys.argv[2])

# %% Define process function


def compute_trajectory_derivatives(pdb_id):
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

    traj = normalize(traj, ca_only=True)
    rmsf = compute_rmsf(traj, normalized=True, ca_only=True)
    # logger.info(f"Worker {os.getpid()} memory after RMSF: {process.memory_info().rss / 1024 / 1024} MB")

    contacts = compute_contacts(
        traj, scheme="ca", ignore_nonprotein=True, normalized=True, ca_only=True
    )
    # logger.info(f"Worker {os.getpid()} memory after contacts: {process.memory_info().rss / 1024 / 1024} MB")

    ca_dist = contacts[0]
    autocorr = compute_autocorrelation(
        traj, precomputed_contacts=contacts, normalized=True, ca_only=True
    )
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


# %% Define all reps
pdb_codes = [pdb_f.stem for pdb_f in pdb_files]
pdb_reps = [(pdb_code, rep) for pdb_code in pdb_codes for rep in range(1, N_REPS + 1)][
    I_START:I_STOP
]
TOTAL_JOBS = len(pdb_reps)

# %% Compute values in parallel
# N_JOBS = 50
# BATCH_SIZE = 1

# logger.info(f"Computing {TOTAL_JOBS} reps in parallel ({N_JOBS} workers, batch size {BATCH_SIZE})")
# results = parallel_pool(pdb_reps, compute_batched_trajectory_derivatives, n_jobs=N_JOBS, batch_size=BATCH_SIZE, report_every=1)
# logger.info(f"Finished processing {len(results)} reps")


# %% Invert Dicts
def invert_dict(l):
    """
    Convert from list of dicts to dict of lists, where the key is the joined "pdb_code" and "rep" keys of the inner dictionary
    """
    out_dict = {}
    keys = l[0].keys()
    for key in keys:
        out_dict[key] = [i[key] for i in l]
    return out_dict


# %% Just do serial to not deal with memory issues
results = []
logger.info(f"Computing {TOTAL_JOBS} reps")
for pdb_id in tqdm(pdb_reps):
    # results.append(compute_trajectory_derivatives(pdb_id))
    results.append(compute_trajectory_derivatives(pdb_id))
# %% # Create HuggingFace dataset

logger.info("Creating HuggingFace dataset")
ds = Dataset.from_dict(invert_dict(results))
logger.info(f"Dataset created with {len(ds)} samples")
out_path = ATLAS_PROCESSED_DATA_DIR / "atlas_derivatives_v2_4162_4167"
logger.info(f"Saving dataset to {out_path}")
ds.save_to_disk(str(out_path))
# logger.info(f"Saved to {out_path}")

# %% Join data sets from different initializations
# logger.info("Joining datasets")

# subsets = [
#     (0, 1000),
#     (1000, 2000),
#     (2000, 3000),
#     (3000, 4000),
#     (4000, 5000),
#     (4162, 4167),
# ]
# all_ds = []
# for s0, s1 in subsets:
#     ds = Dataset.load_from_disk(
#         str(ATLAS_PROCESSED_DATA_DIR / f"atlas_derivatives_v2_{s0}_{s1}")
#     )
#     all_ds.append(ds)

# all_ds = concatenate_datasets(all_ds)
# all_ds.save_to_disk(str(ATLAS_PROCESSED_DATA_DIR / "atlas_derivatives_v2"))
