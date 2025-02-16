# %%
import glob

import torch
from datasets import Dataset
from tqdm import tqdm

from rocketshp import config

# %%
# disbatch_tasks_file = f"{config.PROJ_ROOT}/scripts/01_preprocess/atlas/atlas_shp_disbatch.tasks.txt"

# N_REPS = 3
# ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
# pdb_files = list(ATLAS_DATA_DIR.glob("*/*.pdb"))
# pdb_files = [i for i in pdb_files if ".ca.pdb" not in i.name]
# pdb_codes = [pdb_f.stem for pdb_f in pdb_files]
# pdb_reps = [(pdb_code, rep) for pdb_code in pdb_codes for rep in range(1, N_REPS + 1)]

# disbatch_prefix = "#DISBATCH PREFIX ( cd /mnt/home/ssledzieski/Projects/rocketshp ; module load cudnn; module load nccl; module load cuda; source ~/venvs/scc-rocketshp/bin/activate; python scripts/01_preprocess/atlas/shp_disbatch_single.py "
# disbatch_suffix = "#DISBATCH SUFFIX )"

# with open(disbatch_tasks_file,"w+") as f:
#     f.write(disbatch_prefix + "\n")
#     f.write(disbatch_suffix + "\n")
#     for (pdb_id, rep) in pdb_reps:
#         f.write(f"{pdb_id} {rep}\n")

# %%
# sbatch -n 50 -c 4 disBatch scripts/01_preprocess/atlas/atlas_shp_disbatch.tasks.txt

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


shp_files = glob.glob(f"{config.PROCESSED_DATA_DIR}/atlas/esm_shp/*/*.pt")
results = []
for f in tqdm(shp_files):
    shp = torch.load(f)
    shp["shp"] = shp["shp"].squeeze()
    results.append(shp)
shp_results = invert_dict(results)
ds = Dataset.from_dict(shp_results)
ds.save_to_disk(f"{config.PROCESSED_DATA_DIR}/atlas/esm_shp/dataset")
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
ds = Dataset.load_from_disk(f"{config.PROCESSED_DATA_DIR}/atlas/shp/dataset")

# %%
