# %%
import glob

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm

from rocketshp import config

# %%
disbatch_tasks_file = (
    f"{config.PROJ_ROOT}/scripts/01_preprocess/mdcath/mdcath_shp_fs_disbatch.tasks.txt"
)

# N_REPS = 3
MDCATH_DATA_DIR = config.RAW_DATA_DIR / "mdcath"
mdcath_files = list(MDCATH_DATA_DIR.glob("mdcath_dataset_*.h5"))
N_REPS = 5
TEMPS = [320, 348, 379, 413, 450]
mdcath_reps = [
    (str(mdc_f), rep, temp)
    for mdc_f in mdcath_files
    for rep in range(N_REPS)
    for temp in TEMPS
]

disbatch_prefix = "#DISBATCH PREFIX ( cd /mnt/home/ssledzieski/Projects/rocketshp ; module load cudnn; module load nccl; module load cuda; module load foldseek; source ~/venvs/scc-rocketshp/bin/activate; python scripts/01_preprocess/mdcath/foldseek_shp_disbatch_single.py "
disbatch_suffix = "#DISBATCH SUFFIX )"

with open(disbatch_tasks_file, "w+") as f:
    f.write(disbatch_prefix + "\n")
    f.write(disbatch_suffix + "\n")
    for pdb_id, rep, temp in mdcath_reps:
        f.write(f"{pdb_id} {rep} {temp}\n")

# %%
# sbatch -n 50 -c 4 disBatch scripts/01_preprocess/mdcath/mdcath_shp_fs_disbatch.tasks.txt

# %%
# Compile and save results


def invert_dict(l):
    """
    Convert from list of dicts to dict of lists, where the key is the joined "pdb_code" and "rep" keys of the inner dictionary
    """
    out_dict = {}
    keys = l[0].keys()
    for key in keys:
        out_dict[key] = [i[key] for i in l]
    return out_dict


def convert_to_normalized_shp(preshp, max_dim=20):
    if f.ndim > 2:
        preshp = torch.tensor(preshp).squeeze()
    elif f.ndim == 1:
        preshp = torch.tensor(preshp).unsqueeze(0)
    shp = torch.stack(
        [torch.from_numpy(np.bincount(i, minlength=max_dim)) for i in preshp.T]
    )
    shp = shp.T / shp.sum(axis=1)
    return shp.T


# %%
shp_files = glob.glob(f"{config.PROCESSED_DATA_DIR}/mdcath/fs_shp/*/*.pt")
results = []
for f in tqdm(shp_files):
    shp = torch.load(f)
    shp = shp["fs_shp"].squeeze()
    shp["fs_shp"] = convert_to_normalized_shp(shp, max_dim=20)
    results.append(shp)
shp_results = invert_dict(results)
ds = Dataset.from_dict(shp_results)
ds.save_to_disk(f"{config.PROCESSED_DATA_DIR}/mdcath/fs_shp/dataset")
# %%
# Add to H5 file
# from rocketshp.data.utils import update_h5_dataset

# atlas_processed_h5 = f"{config.PROCESSED_DATA_DIR}/atlas/atlas_processed.h5"
# with open(atlas_processed_h5, "wb") as f:
#     for replicate in tqdm(ds):
#         pdb_id = replicate["pdb_code"]
#         rep = replicate["rep"]
#         update_h5_dataset(f, f"{pdb_id}/R{rep}/shp", replicate)
# %%
