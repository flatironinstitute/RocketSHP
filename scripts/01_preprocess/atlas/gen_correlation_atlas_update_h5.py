import h5py
import torch
from loguru import logger
from tqdm import tqdm

from rocketshp import config
from rocketshp.data.utils import update_h5_dataset

ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
ATLAS_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "atlas"

xtc_files = list(ATLAS_DATA_DIR.glob("*/*.xtc"))
pdb_files = list(ATLAS_DATA_DIR.glob("*/*.pdb"))
pdb_files = [i for i in pdb_files if ".ca.pdb" not in i.name]
N_REPS = 3

STEP = 10
DO_LOCAL_ALIGN = True
local_suff = "local_" if DO_LOCAL_ALIGN else ""

with h5py.File(ATLAS_PROCESSED_DATA_DIR / "atlas_processed.h5", "a") as h5file:
    for pdb_f in tqdm(pdb_files, total=len(pdb_files)):
        pdb_code = pdb_f.stem

        h5file.require_group(pdb_code)

        for rep in range(1, N_REPS + 1):
            xtc_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}_prod_R{rep}_fit.xtc"
            if not xtc_f.exists():
                logger.warning(f"Missing {xtc_f}")
                continue

            gen_correlation_file = str(ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}_{rep}_{local_suff}step{STEP}_corr_matrix.pt")
            gen_correlation = torch.load(gen_correlation_file, weights_only=True)

            update_h5_dataset(h5file, f"{pdb_code}/R{rep}/dyn_corr", gen_correlation)
