# %% # Load packages

import h5py
from loguru import logger
from tqdm import tqdm

from rocketshp import config
from rocketshp.data.utils import update_h5_dataset
from rocketshp.trajectory import compute_generalized_correlation_lmi

# %% Define file paths
logger.info("Defining file paths")
ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
ATLAS_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "atlas"
ATLAS_H5 = ATLAS_PROCESSED_DATA_DIR / "atlas_processed.h5"

xtc_files = list(ATLAS_DATA_DIR.glob("*/*.xtc"))
pdb_files = list(ATLAS_DATA_DIR.glob("*/*.pdb"))
pdb_files = [i for i in pdb_files if ".ca.pdb" not in i.name]
N_REPS = 3
RANDOM_STATE = 42
OVERWRITE_H5 = False

# %% Compute derivatives and store

# build file
with h5py.File(ATLAS_H5, "r+") as h5file:
    # Generalized Dynamic Correlation
    for i, pdb_f in enumerate(tqdm(pdb_files, desc="Computing GCC-LMI")):
        pdb_code = pdb_f.stem
        for rep in range(1, N_REPS + 1):
            xtc_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}_prod_R{rep}_fit.xtc"
            logger.info(f"Processing {pdb_code} rep {rep}")

            gcorr = compute_generalized_correlation_lmi(
                pdb_f,
                xtc_f,
            )

            update_h5_dataset(
                h5file, f"{pdb_code}/R{rep}/gcc_lmi", gcorr, overwrite=True
            )
