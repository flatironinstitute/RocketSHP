import os

import h5py
import torch
from tqdm import tqdm

from rocketshp import config
from rocketshp.datasets.utils import update_h5_dataset

os.environ["LOGURU_LEVEL"] = "INFO"

MDCATH_DATA_DIR = config.RAW_DATA_DIR / "mdcath"
MDCATH_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "mdcath"

mdcath_files = list(MDCATH_DATA_DIR.glob("mdcath_dataset_*.h5"))
TEMPS = [320, 348, 379, 413, 450]
REPS = [0, 1, 2, 3, 4]

STEP = 10
DO_LOCAL_ALIGN = True
local_suff = "local_" if DO_LOCAL_ALIGN else ""

with h5py.File(MDCATH_PROCESSED_DATA_DIR / "mdcath_processed.h5", "a") as h5file:
    for mdc_f in tqdm(mdcath_files, total=len(mdcath_files)):
        pdb_code = mdc_f.stem.split("_")[-1]

        h5file.require_group(pdb_code)

        for temp in TEMPS:
            for rep in REPS:

                gen_correlation_file = str(MDCATH_PROCESSED_DATA_DIR / pdb_code / f"{pdb_code}_{temp}_{rep}_{local_suff}_corr_matrix.pt")
                gen_correlation = torch.load(gen_correlation_file, weights_only=True)

                update_h5_dataset(h5file, f"{pdb_code}/T{temp}/R{rep}/dyn_corr", gen_correlation)
