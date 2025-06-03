from pathlib import Path

import numpy as np
from tqdm import tqdm
from loguru import logger

from rocketshp.trajectory import compute_generalized_correlation_lmi

bioemu_100_root = Path("/mnt/home/ssledzieski/GitHub/bioemu/rshp_mdcath_results_100/")
bioemu_100_results = sorted(bioemu_100_root.glob("*_100"))


for r_100 in tqdm(bioemu_100_results, total=len(bioemu_100_results)):
    k = "_".join(r_100.name.split("_")[:2])

    top_100_path = r_100 / "topology.pdb"
    traj_100_path = r_100 / "samples.xtc"
    save_100_path = r_100 / "gcc_lmi.npy"

    if save_100_path.exists():
        continue

    try:
        gcc_lmi_100 = compute_generalized_correlation_lmi(
            top_100_path, traj_100_path, verbose=False
        )
        np.save(save_100_path, gcc_lmi_100)
    except FileNotFoundError:
        logger.warning(f"File not found: {top_100_path} or {traj_100_path}")
        continue
