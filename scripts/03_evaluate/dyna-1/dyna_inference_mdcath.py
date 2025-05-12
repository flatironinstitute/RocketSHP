# %%
from pathlib import Path

from tqdm import tqdm

num_samples = 10
data_root = Path("/mnt/home/ssledzieski/Projects/rocketshp/data")
output_root = Path("/mnt/home/ssledzieski/GitHub/Dyna-1/rshp_mdcath_results")
output_root.mkdir(parents=True, exist_ok=True)
mdcath_root = data_root / "processed/mdcath"

# %% Get all mdcath
mdcath_dirs = [f for f in mdcath_root.glob("*") if f.is_dir()]
mdcath_pdbs = [x / x.with_suffix(".pdb").name for x in mdcath_dirs]

# %% Create script

with open(f"{output_root}/run_dyna_mdcath.sh", "w") as f:
    f.write("#!/bin/bash\n")
    f.write("mamba activate dyna1\n")
    f.write("cd /mnt/home/ssledzieski/GitHub/Dyna-1\n")

    for pdb_fi in tqdm(mdcath_pdbs):
        pdb_key = pdb_fi.stem
        f.write(f"echo {pdb_key}\n")
        f.write(
            #            f"mkdir -p {output_root}/{pdb_key}; /usr/bin/time -o {output_root}/{pdb_key}/time_log.txt python dyna1.py --pdb {pdb_id} --chain {pdb_chain} --name {pdb_key} --save_dir {output_root}/{pdb_key} --use_pdb_seq --write_to_pdb\n"
            f"mkdir -p {output_root}/{pdb_key}; python dyna1.py --pdb {pdb_fi} --name {pdb_key} --save_dir {output_root}/{pdb_key} --use_pdb_seq --write_to_pdb\n"
        )
# %%
