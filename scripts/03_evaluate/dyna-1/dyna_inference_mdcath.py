#%%
import re
import time
import subprocess as sp
from tqdm import tqdm
from pathlib import Path

num_samples = 10
data_root = Path("/mnt/home/ssledzieski/Projects/rocketshp/data")
output_root = Path("/mnt/home/ssledzieski/GitHub/Dyna-1/rshp_results")
atlas_data = data_root / "raw/atlas"


#%% Get all atlas pdb files
pdb_files = [f for f in atlas_data.glob("**/*.pdb") if f.is_file() and re.search(r"...._.\.pdb", f.name)]

#%% Create script

with open(f"{output_root}/run_dyna.sh", "w") as f:
    f.write("#!/bin/bash\n")
    f.write("mamba activate dyna1\n")
    f.write("cd /mnt/home/ssledzieski/GitHub/Dyna-1\n")

    for pdb_fi in tqdm(pdb_files):
        pdb_key = pdb_fi.stem
        pdb_id = pdb_key.split("_")[0]
        pdb_chain = pdb_key.split("_")[1]
        f.write(f"echo {pdb_key}\n")
        f.write(
#            f"mkdir -p {output_root}/{pdb_key}; /usr/bin/time -o {output_root}/{pdb_key}/time_log.txt python dyna1.py --pdb {pdb_id} --chain {pdb_chain} --name {pdb_key} --save_dir {output_root}/{pdb_key} --use_pdb_seq --write_to_pdb\n"
            f"mkdir -p {output_root}/{pdb_key}; /usr/bin/time -o {output_root}/{pdb_key}/time_log.txt python dyna1.py --pdb {atlas_data / pdb_key[:2] / pdb_key}.pdb --name {pdb_key} --save_dir {output_root}/{pdb_key} --use_pdb_seq --write_to_pdb\n"
        )