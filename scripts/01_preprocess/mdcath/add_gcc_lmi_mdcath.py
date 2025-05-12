# %% # Load packages
import os

import h5py
from loguru import logger
from tqdm import tqdm

from rocketshp import config
from rocketshp.data.mdcath import convert_to_files
from rocketshp.data.utils import update_h5_dataset
from rocketshp.trajectory import compute_generalized_correlation_lmi

# %% Define file paths
logger.info("Defining file paths")
MDCATH_DATA_DIR = config.RAW_DATA_DIR / "mdcath"
MDCATH_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "mdcath"
MDCATH_H5 = MDCATH_PROCESSED_DATA_DIR / "mdcath_processed.h5"

mdcath_files = list(MDCATH_DATA_DIR.glob("mdcath_dataset_*.h5"))
N_REPS = 5
TEMPS = [320, 348, 379, 413, 450]
RANDOM_STATE = 42
OVERWRITE_H5 = False

# %% Compute derivatives and store

# build file
with h5py.File(MDCATH_H5, "r+") as h5file:
    # Generalized Dynamic Correlation
    for i, mdc_f in enumerate(tqdm(mdcath_files, desc="Computing GCC-LMI")):
        pdb_code = str(mdc_f).split("_")[-1].split(".")[0]
        for rep in range(N_REPS):
            for temp in TEMPS:
                if f"{pdb_code}/T{temp}/R{rep}/gcc_lmi" in h5file:
                    continue

                pdb_f, xtc_f = convert_to_files(
                    mdc_f, replica_list=[rep], temp_list=[temp], directory="/tmp/"
                )
                logger.info(f"Processing {pdb_code} rep {rep} temp {temp}")

                # Compute GCC LMI
                gcorr = compute_generalized_correlation_lmi(
                    pdb_f,
                    xtc_f,
                )
                update_h5_dataset(
                    h5file, f"{pdb_code}/T{temp}/R{rep}/gcc_lmi", gcorr, overwrite=True
                )

                # Clean up files
                os.remove(pdb_f)
                os.remove(xtc_f)
