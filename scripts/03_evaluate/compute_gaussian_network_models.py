# %% Imports
import os

import numpy as np
from prody import GNM, parsePDB
from tqdm import tqdm

from rocketshp import config

# %% File paths
ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
pdb_files = list(ATLAS_DATA_DIR.glob("*/*.pdb"))
pdb_files = [i for i in pdb_files if "ca" not in i.stem]

N_MODES = 50
DIST_CUTOFF = 7.0

# %% Compute GNMs
for pdb_f in tqdm(pdb_files):
    os.makedirs(
        f"{config.PROCESSED_DATA_DIR}/atlas/gaussian_net_models/{pdb_f.stem[:2]}",
        exist_ok=True,
    )
    struct = parsePDB(str(pdb_f))
    calphas = struct.select("calpha")
    gnm = GNM(f"{pdb_f.stem} GNM")
    gnm.buildKirchhoff(calphas, cutoff=DIST_CUTOFF)
    gnm.calcModes(n_modes=N_MODES)
    eigvals = gnm.getEigvals()
    eigvecs = gnm.getEigvecs()
    covar = gnm.getCovariance()
    f_root = f"{config.PROCESSED_DATA_DIR}/atlas/gaussian_net_models/{pdb_f.stem[:2]}/{pdb_f.stem}_gnm"
    np.savez(f_root, eigvals=eigvals, eigvecs=eigvecs, covar=covar)
    # print(f"Computed GNM for {pdb_f.stem}")
# %%
